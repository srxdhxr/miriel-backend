from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from shared.database import get_db
from shared.models import ScrapedDocument
from sqlmodel import select
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json
import os
from typing import Optional, List
import logging
import requests
import base64
from io import BytesIO
import tarfile
import torch
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self, domain_id: Optional[int] = None):
        """Initialize manager for a specific domain or all domains"""
        logger.info("Initializing VectorStoreManager...")
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt_tab',quiet = True)
            
            self.stop_words = set(stopwords.words('english'))
            self.domain_id = domain_id
            
            # Configure embeddings model
            self.embeddings = self._get_embeddings()
            logger.info("Embeddings model initialized")
            
            # Get Ollama API URL from environment with fallback
            self.ollama_api_url = os.getenv('OLLAMA_API_URL', 'http://ollama:11434')
            logger.info(f"Using Ollama API URL: {self.ollama_api_url}")
            
            # Load and process documents
            self.documents = self.load_documents_from_db()
            logger.info(f"Loaded {len(self.documents)} documents")
            
            # Create vector store
            self.vectorstore = self.create_vectorstore()
            logger.info("Vector store created successfully")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def verify_ollama_connection(self):
        """Verify connection to Ollama service"""
        try:
            # Try the root endpoint of the ollama-rag service
            response = requests.get(f"{self.ollama_api_url}/")
            if response.status_code == 200:
                logger.info("Successfully connected to Ollama RAG service")
                return True
            else:
                logger.error(f"Failed to connect to Ollama RAG service: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama RAG service: {str(e)}")
            return False

    def _get_embeddings(self):
        """Initialize efficient embeddings model"""
        try:
            # Using MiniLM-L6-v2 for efficiency and good performance
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise

    def get_vector_path(self) -> str:
        """Get path for vector store based on domain"""
        base_path = "vectorstore"
        if self.domain_id:
            return f"{base_path}/domain_{self.domain_id}"
        return base_path

    def normalize_content(self, text: str) -> str:
        """Normalize content while preserving important information"""
        try:
            # Basic cleaning
            text = text.lower().strip()
            
            # Remove URLs but keep domain names
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove special characters but keep important punctuation
            text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
            
            # Tokenize and remove stopwords
            words = word_tokenize(text)
            filtered_words = [word for word in words if word not in self.stop_words or word in ['.', ',', '!', '?']]
            
            # Rejoin with proper spacing
            return ' '.join(filtered_words)
        except Exception as e:
            logger.error(f"Error normalizing content: {e}")
            return text

    def load_documents_from_db(self) -> List[Document]:
        """Load and process documents efficiently"""
        logger.info(f"Loading documents for {'domain ' + str(self.domain_id) if self.domain_id else 'all domains'}")
        documents = []
        
        try:
            with get_db() as db:
                # Build query based on domain_id
                stmt = select(ScrapedDocument)
                if self.domain_id:
                    stmt = stmt.where(ScrapedDocument.domain_id == self.domain_id)
                
                results = db.exec(stmt).all()
                
                # Process documents in batches
                batch_size = 100
                for i in range(0, len(results), batch_size):
                    batch = results[i:i + batch_size]
                    for doc in batch:
                        normalized_content = self.normalize_content(doc.content_raw)
                        if normalized_content.strip():  # Skip empty documents
                            langchain_doc = Document(
                                page_content=normalized_content,
                                metadata={
                                    "source": doc.url,
                                    "domain_id": doc.domain_id,
                                    "last_scraped": doc.last_scraped.isoformat()
                                }
                            )
                            documents.append(langchain_doc)
            
            # Split documents with optimal settings
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunks for better retrieval
                chunk_overlap=50,  # Minimal overlap
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " "]
            )
            
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Created {len(split_docs)} text chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def create_vectorstore(self):
        """Create optimized FAISS vectorstore"""
        try:
            logger.info("Creating vector store...")
            return FAISS.from_documents(
                self.documents,
                self.embeddings,
                normalize_L2=True
            )
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    def compress_vectorstore(self, path: str) -> bytes:
        """Compress vectorstore efficiently"""
        try:
            tar_buffer = BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode='w:gz', compresslevel=6) as tar:
                tar.add(path, arcname=os.path.basename(path))
            return tar_buffer.getvalue()
        except Exception as e:
            logger.error(f"Error compressing vector store: {e}")
            raise

    def save_vectorstore(self):
        """Save and upload vector store"""
        try:
            print("DEBUG: Starting save_vectorstore method")
            logger.info("DEBUG: Starting save_vectorstore method")
            
            try:
                # Save locally first
                local_path = self.get_vector_path()
                print(f"DEBUG: Local path is {local_path}")
                logger.info(f"DEBUG: Local path is {local_path}")
                
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"DEBUG: Created directory at {os.path.dirname(local_path)}")
                logger.info(f"DEBUG: Created directory at {os.path.dirname(local_path)}")
                
                self.vectorstore.save_local(local_path)
                print(f"DEBUG: Vector store saved locally to {local_path}")
                logger.info(f"DEBUG: Vector store saved locally to {local_path}")

                print("DEBUG: Local save completed, proceeding to Ollama verification")
                logger.info("DEBUG: Local save completed, proceeding to Ollama verification")
            except Exception as e:
                print(f"DEBUG: Error in local save step: {str(e)}")
                logger.error(f"DEBUG: Error in local save step: {str(e)}")
                raise

            try:
                print("DEBUG: About to verify Ollama connection")
                logger.info("DEBUG: About to verify Ollama connection")
                # Verify Ollama connection before attempting upload
                connection_result = self.verify_ollama_connection()
                print(f"DEBUG: Ollama connection result: {connection_result}")
                logger.info(f"DEBUG: Ollama connection result: {connection_result}")
                
                if not connection_result:
                    print("DEBUG: Ollama connection failed, using fallback")
                    logger.warning("DEBUG: Ollama connection failed, using fallback")
            except Exception as e:
                print(f"DEBUG: Error in Ollama connection verification: {str(e)}")
                logger.error(f"DEBUG: Error in Ollama connection verification: {str(e)}")
                print(f"DEBUG: Full exception details: {repr(e)}")
                logger.error(f"DEBUG: Full exception details: {repr(e)}")
                raise

            if not connection_result:
                try:
                    # Fallback to direct file system storage
                    shared_vector_path = "/app/vectorstore"
                    if self.domain_id:
                        shared_vector_path = f"{shared_vector_path}/domain_{self.domain_id}"
                    
                    print(f"DEBUG: Using shared vector path: {shared_vector_path}")
                    logger.info(f"DEBUG: Using shared vector path: {shared_vector_path}")
                    
                    # Ensure directory exists
                    os.makedirs(shared_vector_path, exist_ok=True)
                    print(f"DEBUG: Created shared directory at {shared_vector_path}")
                    logger.info(f"DEBUG: Created shared directory at {shared_vector_path}")
                    
                    # Save vector store and metadata
                    self.vectorstore.save_local(shared_vector_path)
                    print(f"DEBUG: Saved vector store to {shared_vector_path}")
                    logger.info(f"DEBUG: Saved vector store to {shared_vector_path}")
                    
                    # Check for expected files
                    expected_files = ['index.faiss', 'index.pkl']
                    for file in expected_files:
                        file_path = os.path.join(shared_vector_path, file)
                        exists = os.path.exists(file_path)
                        print(f"DEBUG: Checking {file}: {'exists' if exists else 'missing'}")
                        logger.info(f"DEBUG: Checking {file}: {'exists' if exists else 'missing'}")
                        if exists:
                            size = os.path.getsize(file_path)
                            print(f"DEBUG: {file} size: {size} bytes")
                            logger.info(f"DEBUG: {file} size: {size} bytes")
                    
                    # List directory contents after save
                    print(f"DEBUG: Contents of {shared_vector_path}:")
                    logger.info(f"DEBUG: Contents of {shared_vector_path}:")
                    for item in os.listdir(shared_vector_path):
                        file_path = os.path.join(shared_vector_path, item)
                        size = os.path.getsize(file_path)
                        print(f"DEBUG: Found file: {item} (size: {size} bytes)")
                        logger.info(f"DEBUG: Found file: {item} (size: {size} bytes)")
                    
                    # Save metadata separately
                    metadata = {
                        "domain_id": self.domain_id,
                        "document_count": len(self.documents),
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    metadata_path = f"{shared_vector_path}/metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f)
                    print(f"DEBUG: Saved metadata to {metadata_path}")
                    logger.info(f"DEBUG: Saved metadata to {metadata_path}")
                    
                except Exception as e:
                    print(f"DEBUG: Error in fallback save process: {str(e)}")
                    logger.error(f"DEBUG: Error in fallback save process: {str(e)}")
                    raise
                
                logger.info(f"Vector store saved to shared volume at {shared_vector_path}")
                return

            try:
                print("DEBUG: Starting vector compression")
                logger.info("DEBUG: Starting vector compression")
                # Compress and encode vector store
                compressed_data = self.compress_vectorstore(local_path)
                print(f"DEBUG: Compressed data size: {len(compressed_data)} bytes")
                logger.info(f"DEBUG: Compressed data size: {len(compressed_data)} bytes")
                
                encoded_data = base64.b64encode(compressed_data).decode('utf-8')
                print("DEBUG: Data encoded successfully")
                logger.info("DEBUG: Data encoded successfully")

                # Prepare metadata
                metadata = {
                    "domain_id": self.domain_id,
                    "document_count": len(self.documents),
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "timestamp": datetime.utcnow().isoformat()
                }
                print(f"DEBUG: Prepared metadata: {metadata}")
                logger.info(f"DEBUG: Prepared metadata: {metadata}")

                # Send to Ollama RAG service
                print(f"DEBUG: Attempting to upload to {self.ollama_api_url}/upload_vectors")
                logger.info(f"DEBUG: Attempting to upload to {self.ollama_api_url}/upload_vectors")
                
                response = requests.post(
                    f"{self.ollama_api_url}/upload_vectors",
                    json={
                        "vector_data": encoded_data,
                        "metadata": metadata
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=300
                )
                print(f"DEBUG: Upload response status: {response.status_code}")
                logger.info(f"DEBUG: Upload response status: {response.status_code}")
                
                if response.status_code != 200:
                    print(f"DEBUG: Upload failed with status {response.status_code}, response: {response.text}")
                    logger.warning(f"DEBUG: Upload failed with status {response.status_code}, response: {response.text}")
                    # Save to the shared volume that ollama-rag can access
                    shared_vector_path = "/app/vectorstore"
                    if self.domain_id:
                        shared_vector_path = f"{shared_vector_path}/domain_{self.domain_id}"
                    
                    print(f"DEBUG: Falling back to direct storage at {shared_vector_path}")
                    logger.info(f"DEBUG: Falling back to direct storage at {shared_vector_path}")
                    
                    # Ensure directory exists
                    os.makedirs(shared_vector_path, exist_ok=True)
                    
                    # Save vector store and metadata
                    self.vectorstore.save_local(shared_vector_path)
                    print("DEBUG: Vector store saved to shared volume")
                    logger.info("DEBUG: Vector store saved to shared volume")
                    
                    # Save metadata separately
                    with open(f"{shared_vector_path}/metadata.json", 'w') as f:
                        json.dump(metadata, f)
                    print("DEBUG: Metadata saved to shared volume")
                    logger.info("DEBUG: Metadata saved to shared volume")
                    
                    logger.info(f"Vector store saved to shared volume at {shared_vector_path}")
                else:
                    print("DEBUG: Vector store successfully uploaded via API")
                    logger.info("DEBUG: Vector store successfully uploaded via API")

            except Exception as e:
                print(f"DEBUG: Error during compression/upload: {str(e)}")
                logger.error(f"DEBUG: Error during compression/upload: {str(e)}")
                print(f"DEBUG: Full exception details: {repr(e)}")
                logger.error(f"DEBUG: Full exception details: {repr(e)}")
                
                # Fallback to direct storage
                try:
                    shared_vector_path = "/app/vectorstore"
                    if self.domain_id:
                        shared_vector_path = f"{shared_vector_path}/domain_{self.domain_id}"
                    
                    print(f"DEBUG: Exception occurred, falling back to direct storage at {shared_vector_path}")
                    logger.info(f"DEBUG: Exception occurred, falling back to direct storage at {shared_vector_path}")
                    
                    # Ensure directory exists
                    os.makedirs(shared_vector_path, exist_ok=True)
                    
                    # Save vector store and metadata
                    self.vectorstore.save_local(shared_vector_path)
                    print("DEBUG: Vector store saved to shared volume")
                    logger.info("DEBUG: Vector store saved to shared volume")
                    
                    # Save metadata separately
                    with open(f"{shared_vector_path}/metadata.json", 'w') as f:
                        json.dump(metadata, f)
                    print("DEBUG: Metadata saved to shared volume")
                    logger.info("DEBUG: Metadata saved to shared volume")
                    
                    logger.info(f"Vector store saved to shared volume at {shared_vector_path}")
                except Exception as inner_e:
                    print(f"DEBUG: Error during fallback save: {str(inner_e)}")
                    logger.error(f"DEBUG: Error during fallback save: {str(inner_e)}")
                    raise

        except Exception as e:
            logger.error(f"Error in save_vectorstore: {str(e)}")
            raise

    @staticmethod
    def load_vectorstore(domain_id: Optional[int] = None) -> Optional[FAISS]:
        """Load vector store with efficient settings"""
        try:
            path = f"vectorstore/domain_{domain_id}" if domain_id else "vectorstore"
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
            
            vectorstore = FAISS.load_local(
                path,
                embeddings,
                normalize_L2=True,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded successfully from {path}")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None

    def update_vectors(self):
        """Update vectors efficiently"""
        try:
            logger.info("Starting vector update process")
            
            # Verify Ollama connection
            if not self.verify_ollama_connection():
                logger.error("Cannot update vectors: Ollama service unavailable")
                return

            path = self.get_vector_path()
            if os.path.exists(path):
                existing_vectorstore = self.load_vectorstore(self.domain_id)
                if existing_vectorstore:
                    self.vectorstore = existing_vectorstore
                    logger.info("Loaded existing vector store")

            new_documents = self.load_documents_from_db()
            if new_documents:
                self.vectorstore.add_documents(new_documents)
                logger.info(f"Added {len(new_documents)} new documents to vector store")
                self.save_vectorstore()
                logger.info("Updated vector store saved and uploaded")
            else:
                logger.info("No new documents to add")
            
        except Exception as e:
            logger.error(f"Error updating vectors: {str(e)}")
            raise 