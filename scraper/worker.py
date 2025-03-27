from fastapi import FastAPI, HTTPException
from .scraper import WebsiteScraper
from shared.database import get_db
from shared.models import Domain, ScrapedDocument
from datetime import datetime
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

@app.post("/scrape/{domain_id}")
async def scrape_domain(domain_id: int):
    """Endpoint to scrape a domain"""
    try:
        logger.info(f"Received scraping request for domain {domain_id}")
        # Get domain URL from database
        with get_db() as db:
            domain = db.get(Domain, domain_id)
            if not domain:
                logger.error(f"Domain {domain_id} not found")
                raise HTTPException(status_code=404, detail="Domain not found")
            
            domain_url = domain.domain_url
            logger.info(f"Found domain URL: {domain_url}")
        
        # Initialize scraper and run using async method
        logger.info(f"Starting scraping for domain {domain_id}")
        scraper = WebsiteScraper(domain_id=domain_id, domain_url=domain_url)
        scraped_content = await scraper.scrape_website()  # Use the async version
        
        # Save results to database
        logger.info(f"Saving {len(scraped_content)} pages to database")
        with get_db() as db:
            for url, data in scraped_content.items():
                doc = ScrapedDocument(
                    url=url,
                    url_hash=data['url_hash'],
                    content_raw=data['content_raw'],
                    domain_id=domain_id,
                    last_scraped=datetime.fromisoformat(data['last_scraped'])
                )
                db.add(doc)
            db.commit()
        
        logger.info(f"Scraping completed for domain {domain_id}")
        return {
            "status": "success",
            "message": f"Scrape completed for domain {domain_id}",
            "pages_scraped": len(scraped_content)
        }
    except Exception as e:
        logger.error(f"Error scraping domain {domain_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting scraper service")
    uvicorn.run(app, host="0.0.0.0", port=8001) 