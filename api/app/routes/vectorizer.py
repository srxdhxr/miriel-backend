from fastapi import APIRouter, HTTPException
import httpx
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vectorize", tags=["vectorizer"])

VECTORIZER_SERVICE_URL = os.getenv('VECTORIZER_SERVICE_URL', 'http://vectorizer:8002')

@router.post("/{domain_id}")
async def vectorize_domain(domain_id: int) -> Dict[str, Any]:
    """Trigger vectorization via HTTP request to vectorizer service"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(f"{VECTORIZER_SERVICE_URL}/vectorize/{domain_id}")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error calling vectorizer service: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vectorizer service error: {str(e)}") 