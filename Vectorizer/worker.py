from fastapi import FastAPI, HTTPException
from Vectorizer.vector_store import VectorStoreManager
from Vectorizer.vector_tasks import process_vectorization
from shared.database import get_db
from shared.models import Domain
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.post("/vectorize/{domain_id}")
async def vectorize_domain(domain_id: int):
    """Endpoint to vectorize content for a domain"""
    try:
        logger.info(f"Received vectorization request for domain {domain_id}")
        
        # Check if domain exists
        with get_db() as db:
            domain = db.get(Domain, domain_id)
            if not domain:
                logger.error(f"Domain {domain_id} not found")
                raise HTTPException(status_code=404, detail="Domain not found")
        
        # Process vectorization directly
        logger.info(f"Starting vectorization for domain {domain_id}")
        result = process_vectorization(domain_id)
        
        if result["status"] == "error":
            logger.error(f"Error in vectorization: {result['message']}")
            raise HTTPException(status_code=500, detail=result["message"])
        
        logger.info(f"Vectorization completed for domain {domain_id}")
        return {
            "status": "success",
            "message": f"Vectorization completed for domain {domain_id}",
            "documents_processed": result.get("documents_processed", 0)
        }
    except Exception as e:
        logger.error(f"Error vectorizing domain {domain_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting vectorizer service")
    uvicorn.run(app, host="0.0.0.0", port=8000) 