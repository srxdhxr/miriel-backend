from fastapi import APIRouter
from typing import Dict

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"} 