from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import os
from typing import Dict, List, Optional
import logging
import sys
import torch

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Check GPU availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Define a custom prompt template
CUSTOM_PROMPT = """You are a helpful assistant for Alliant Credit Union. Use the following pieces of context to answer questions about Alliant's products and services. If you don't know the answer based on the provided context, just say that you don't know, don't try to make up an answer.

When answering questions about credit cards or financial products:
1. Focus on the specific features and benefits mentioned in the context
2. Include any eligibility requirements if mentioned
3. Mention interest rates or rewards if available in the context
4. Include any special offers or promotions if mentioned

Context: {context}

Question: {question}

Please provide a detailed answer based only on the context provided above. If specific details are not mentioned in the context, indicate that those details are not available rather than making assumptions."""

class RAGSystem:
    def __init__(self, domain_id: int = 1):
        """Initialize RAG system with domain-specific vector store"""
        logger.info(f"Initializing RAG system for domain {domain_id}")
        self.domain_id = domain_id
        self.vector_store_path = f"/app/vector_store/domain_{domain_id}"
        
        try:
            # Initialize embeddings
            logger.debug("Initializing embeddings model")
            self.embeddings = self._get_embeddings()
            logger.info("Embeddings model initialized successfully")
            
            # Load vector store
            logger.debug("Loading vector store")
            self.vector_store = self._load_vector_store()
            logger.info("Vector store loaded successfully")
            
            # Initialize Ollama LLM with GPU support
            logger.debug("Initializing LLM")
            self.llm = Ollama(
                model="mistral:7b-instruct-v0.2-q3_K_S",
                temperature=0.3,
                top_k=50,
                top_p=0.95,
                repeat_penalty=1.2,
                num_ctx=2048,
                num_gpu=1  # Enable GPU usage
            )
            logger.info("LLM initialized successfully with GPU support")
            
            # Create RetrievalQA chain
            self.qa_chain = self._create_qa_chain()
            logger.info("QA chain created successfully")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise
        
    def _get_embeddings(self):
        """Initialize efficient embeddings model with GPU support"""
        try:
            # Using all-MiniLM-L6-v2 with GPU acceleration
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': DEVICE},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32,
                    'device': DEVICE
                }
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
            
    def _load_vector_store(self):
        """Load the FAISS vector store"""
        try:
            logger.debug(f"Attempting to load vector store from: {self.vector_store_path}")
            logger.debug(f"Checking for files: index.faiss, index.pkl, metadata.json")
            
            if not os.path.exists(self.vector_store_path):
                logger.error(f"Vector store directory not found at {self.vector_store_path}")
                raise FileNotFoundError(f"Vector store not found for domain {self.domain_id}")
            
            # Check individual files
            required_files = ['index.faiss', 'index.pkl', 'metadata.json']
            for file in required_files:
                file_path = os.path.join(self.vector_store_path, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    logger.debug(f"Found {file} (size: {file_size} bytes)")
                else:
                    logger.error(f"Required file {file} not found")
                    raise FileNotFoundError(f"Required file {file} not found in vector store")
            
            logger.debug("Loading vector store with FAISS...")
            vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                normalize_L2=True  # Better similarity search
            )
            
            logger.debug(f"Vector store loaded successfully. Index size: {vector_store.index.ntotal}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
            
    def _create_qa_chain(self):
        """Create an optimized RetrievalQA chain"""
        try:
            # Create prompt template
            prompt = PromptTemplate(
                template=CUSTOM_PROMPT,
                input_variables=["context", "question"]
            )
            
            # Create retriever with improved search kwargs
            retriever = self.vector_store.as_retriever(
                search_type="similarity",  # Using similarity search
                search_kwargs={
                    "k": 6,  # Get top 6 documents
                    "score_threshold": 0.1,  # Lower threshold to see more results
                }
            )
            
            # Create QA chain with modified settings
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Using stuff method for combining documents
                retriever=retriever,
                return_source_documents=True,
                verbose=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "verbose": True,
                    "document_separator": "\n\n"  # Clear separation between documents
                }
            )
            
            # Log the retriever configuration
            logger.debug(f"Retriever configured with search_type: similarity")
            logger.debug(f"Search parameters: k=6, score_threshold=0.1")
            
            return qa_chain  # Return the QA chain directly
            
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            raise
    
    async def get_response(self, query: str, temperature: float = 0.7) -> Dict:
        """Get response using RAG approach"""
        try:
            logger.debug(f"Processing query: {query}")
            
            # Log query embedding process
            logger.debug("Generating query embeddings...")
            query_embedding = self.embeddings.embed_query(query)
            logger.debug(f"Query embedding shape: {len(query_embedding)}")
            
            # Update temperature if needed
            self.llm.temperature = temperature
            
            # Get response from QA chain
            logger.debug("Calling QA chain...")
            
            # First try to get similar documents directly from vector store
            logger.debug("Searching vector store directly...")
            similar_docs = self.vector_store.similarity_search_with_score(query, k=6)
            if similar_docs:
                logger.debug("Found similar documents directly:")
                for i, (doc, score) in enumerate(similar_docs):
                    logger.debug(f"Doc {i+1} Score: {score}")
                    logger.debug(f"Content preview: {doc.page_content[:200]}...")
            
            result = await self.qa_chain.ainvoke({
                "query": query
            })
            logger.debug(f"Raw QA chain result: {result}")
            
            # Log retrieved documents for debugging
            logger.debug("Retrieved documents:")
            total_docs = len(result.get("source_documents", []))
            logger.debug(f"Total documents retrieved: {total_docs}")
            
            for idx, doc in enumerate(result.get("source_documents", []), 1):
                logger.debug(f"\nDocument {idx}/{total_docs}:")
                logger.debug(f"Similarity score: {doc.metadata.get('score', 0)}")
                logger.debug(f"Source URL: {doc.metadata.get('source', 'Unknown')}")
                logger.debug(f"Last scraped: {doc.metadata.get('last_scraped', 'Unknown')}")
                logger.debug(f"Content preview: {doc.page_content[:200]}...")
                logger.debug("-" * 80)
            
            # Extract sources from source documents
            sources = []
            if result.get("source_documents"):
                for doc in result["source_documents"]:
                    source_info = {
                        "url": doc.metadata.get("source", "Unknown source"),
                        "last_scraped": doc.metadata.get("last_scraped", "Unknown"),
                        "domain_id": doc.metadata.get("domain_id", self.domain_id),
                        "score": doc.metadata.get("score", 0)
                    }
                    sources.append(source_info)
            
            # Get the response text directly from the result field
            response_text = result.get("result", None)
            if not response_text:
                logger.warning("No response text found in result")
                logger.debug(f"Available fields in result: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            else:
                logger.debug(f"Found response text in result: {response_text[:200]}...")
            
            response = {
                "response": response_text.strip() if response_text else None,
                "error": None,
                "sources": sources
            }
            
            # Log response and sources
            logger.debug(f"Generated response: {response['response']}")
            logger.debug(f"Number of sources found: {len(sources)}")
            if not sources:
                logger.warning(f"No sources found for query: {query}")
                logger.debug("Vector store statistics:")
                logger.debug(f"Total documents in store: {self.vector_store.index.ntotal}")
                logger.debug(f"Vector dimension: {self.vector_store.index.d}")
            
            return response
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                "response": "I apologize, but I encountered an error while processing your question.",
                "error": str(e),
                "sources": []
            } 