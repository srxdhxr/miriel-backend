from Vectorizer.vector_store import VectorStoreManager
from shared.database import get_db
from shared.models import Domain
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def process_vectorization(domain_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Redis queue task to process vectorization pipeline:
    1. Load raw content
    2. Normalize content
    3. Create vectors
    4. Save to disk
    
    Args:
        domain_id: Optional domain ID. If None, processes all domains
    """
    try:
        logger.info(f"Starting vectorization task for {'domain ' + str(domain_id) if domain_id else 'all domains'}")
        
        # Validate domain if specified
        if domain_id:
            with get_db() as db:
                domain = db.get(Domain, domain_id)
                if not domain:
                    error_msg = f"Domain {domain_id} not found"
                    logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
        
        # Initialize vector store manager
        vector_manager = VectorStoreManager(domain_id=domain_id)
        
        # Process pipeline
        try:
            # Load and normalize documents (handled in VectorStoreManager initialization)
            doc_count = len(vector_manager.documents)
            logger.info(f"Loaded and normalized {doc_count} documents")
            
            # Create and save vector store
            vector_manager.save_vectorstore()
            logger.info("Vector store created and saved successfully")
            
            return {
                "status": "success",
                "message": f"Vectorization completed for {'domain ' + str(domain_id) if domain_id else 'all domains'}",
                "documents_processed": doc_count,
                "domain_id": domain_id
            }
            
        except Exception as e:
            error_msg = f"Error during vectorization pipeline: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
            
    except Exception as e:
        error_msg = f"Error in vectorization process: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

def update_vectors(domain_id: int) -> Dict[str, Any]:
    """
    Redis queue task to update existing vectors:
    1. Load existing vectors
    2. Process new/updated content
    3. Update vector store
    4. Save to disk
    
    Args:
        domain_id: Domain ID to update
    """
    try:
        logger.info(f"Starting vector update task for domain {domain_id}")
        
        # Validate domain
        with get_db() as db:
            domain = db.get(Domain, domain_id)
            if not domain:
                error_msg = f"Domain {domain_id} not found"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
        
        # Initialize vector store manager and update
        try:
            vector_manager = VectorStoreManager(domain_id=domain_id)
            vector_manager.update_vectors()
            
            return {
                "status": "success",
                "message": f"Vector store updated for domain {domain_id}",
                "documents_processed": len(vector_manager.documents),
                "domain_id": domain_id
            }
            
        except Exception as e:
            error_msg = f"Error updating vectors: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
            
    except Exception as e:
        error_msg = f"Error in update process: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

def post_scrape_vectorization(domain_id: int) -> Dict[str, Any]:
    """
    Redis queue task to run after scraping completes:
    1. Check if vectors exist
    2. Create or update vectors accordingly
    
    Args:
        domain_id: Domain ID that was just scraped
    """
    try:
        logger.info(f"Starting post-scrape vectorization for domain {domain_id}")
        
        # Check if vectors already exist
        vector_store = VectorStoreManager.load_vectorstore(domain_id)
        
        if vector_store:
            # Update existing vectors
            return update_vectors(domain_id)
        else:
            # Create new vectors
            return process_vectorization(domain_id)
            
    except Exception as e:
        error_msg = f"Error in post-scrape vectorization: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
