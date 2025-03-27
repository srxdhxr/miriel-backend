import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch
import pickle
import json
import requests
import os
from typing import Dict, List, Optional
import logging
from datetime import datetime
import aiohttp
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastRAG:
    def __init__(self, domain_id: int = 1, base_path: str = "/app"):
        """Initialize RAG system with existing FAISS index
        
        Args:
            domain_id: Domain ID to load correct vector store
            base_path: Base path for vector store (default: /app)
        """
        self.domain_id = domain_id
        self.vector_store_path = os.path.join(base_path, "vectorstore", f"domain_{domain_id}")
        
        # Initialize embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.embed_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            device=self.device
        )
        
        # Initialize HTTP session for Ollama
        self.session = None
        
        # Load FAISS index and docstore
        self._load_vector_store()
        
    async def initialize(self):
        """Initialize async resources"""
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        """Cleanup async resources"""
        if self.session:
            await self.session.close()
            self.session = None
        
    def _load_vector_store(self):
        """Load FAISS index and document store from disk"""
        try:
            # Load FAISS index
            index_path = os.path.join(self.vector_store_path, "index.faiss")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found at {index_path}")
                
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Move to GPU if available
            if self.device == "cuda":
                logger.info("Starting GPU index transfer...")
                try:
                    logger.info("Creating GPU resources...")
                    res = faiss.StandardGpuResources()
                    logger.info("GPU resources created successfully")
                    
                    logger.info("Transferring index to GPU...")
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logger.info("Index successfully transferred to GPU")
                except Exception as e:
                    logger.error(f"Error during GPU transfer: {str(e)}")
                    logger.warning("Falling back to CPU index")
            else:
                logger.info("Using CPU index as CUDA is not available")
            
            # Load docstore and UUID mapping
            logger.info("Loading docstore and UUID mapping...")
            metadata_path = os.path.join(self.vector_store_path, "index.pkl")
            if not os.path.exists(metadata_path):
                logger.error(f"Docstore not found at path: {metadata_path}")
                raise FileNotFoundError(f"Docstore not found at {metadata_path}")
            
            logger.info(f"Found metadata file at {metadata_path}")
            
            with open(metadata_path, 'rb') as f:
                self.docstore, self.id_to_uuid = pickle.load(f)
                
            logger.info(f"Loaded docstore with {len(self.id_to_uuid)} documents")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    def search(self, query: str, k: int = 2) -> List[Dict]:
        """Search for similar documents"""
        # Generate query embedding
        query_embedding = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Search in FAISS
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            k
        )
        
        # Get matching documents with scores
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.id_to_uuid):
                uuid = self.id_to_uuid[idx]
                doc = self.docstore.search(uuid)
                
                # Extract metadata in the format expected by the API
                metadata = doc.metadata or {}
                
                # Store score separately as it's not part of the API model
                score = float(1 / (1 + distance))
                
                results.append({
                    "content": doc.page_content,
                    "source": {
                        "url": metadata.get("source", "Unknown source"),
                        "last_scraped": metadata.get("last_scraped", datetime.now().isoformat()),
                        "domain_id": self.domain_id
                    },
                    "score": score
                })
        
        return results

    async def get_response(self, query: str, temperature: float = 0.7) -> Dict:
        """Get response using RAG approach"""
        try:
            logger.info("Starting RAG response generation")
            # Get relevant documents
            logger.info("Searching for relevant documents")
            relevant_docs = self.search(query)
            logger.info(f"Found {len(relevant_docs)} relevant documents")
            
            if not relevant_docs:
                logger.info("No relevant documents found")
                return {
                    "response": "No relevant information found.",
                    "error": None,
                    "sources": []
                }
            
            # Prepare context from document content
            context = "\n\n".join([doc["content"] for doc in relevant_docs])
            logger.info(f"Prepared context with {len(context)} characters")
            
            # Prepare prompt
            prompt = f"""You are a helpful assistant. Use the following context to answer the question. If you don't know the answer based on the context, just say that you don't know.

Context: {context}

Question: {query}

Answer: """
            logger.info("Prepared prompt for Ollama")

            # Ensure session exists
            if not self.session:
                logger.info("Initializing aiohttp session")
                await self.initialize()

            try:
                logger.info("Attempting to connect to Ollama")
                # Call Ollama API using aiohttp with timeout
                async with self.session.post(
                    'http://localhost:11434/api/generate',  # Using localhost since Ollama is in the same container
                    json={
                        "model": "mistral:7b-instruct-v0.2-q3_K_S",
                        "prompt": prompt,
                        "temperature": temperature,
                        "top_k": 50,
                        "top_p": 0.95,
                        "repeat_penalty": 1.2,
                        "stream": True  # Ensure streaming is enabled
                    },
                    timeout=aiohttp.ClientTimeout(total=60)  # 60 second timeout
                ) as response:
                    logger.info("Connected to Ollama, starting to read response")
                    # Process streaming response
                    answer = ""
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    answer += data["response"]
                                    logger.debug("Received response chunk")
                                if data.get("done", False):
                                    logger.info("Ollama response complete")
                                    break
                            except json.JSONDecodeError:
                                logger.error(f"Failed to decode JSON from line: {line}")
                                continue

                logger.info("Formatting final response")
                # Format response to match API's expected structure
                return {
                    "response": answer.strip() if answer else "Error: No response generated",
                    "error": None,
                    "sources": [
                        {
                            "url": doc["source"]["url"],
                            "last_scraped": doc["source"]["last_scraped"],
                            "domain_id": doc["source"]["domain_id"]
                        }
                        for doc in relevant_docs
                    ]
                }
            except asyncio.TimeoutError:
                logger.error("Request to Ollama timed out")
                return {
                    "response": "I apologize, but the request timed out while waiting for a response.",
                    "error": "Request timeout",
                    "sources": []
                }
            except aiohttp.ClientError as e:
                logger.error(f"Connection error with Ollama: {str(e)}")
                return {
                    "response": "I apologize, but I encountered a connection error with the language model.",
                    "error": str(e),
                    "sources": []
                }

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question.",
                "error": str(e),
                "sources": []
            } 