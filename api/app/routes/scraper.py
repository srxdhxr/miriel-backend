from fastapi import APIRouter, HTTPException
import httpx
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scrape", tags=["scraper"])

SCRAPER_SERVICE_URL = os.getenv('SCRAPER_SERVICE_URL', 'http://scraper:8001')

@router.post("/{domain_id}")
async def scrape_domain(domain_id: int) -> Dict[str, Any]:
    """Trigger scraping via HTTP request to scraper service"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{SCRAPER_SERVICE_URL}/scrape/{domain_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error calling scraper service: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Scraper service error: {str(e)}") 