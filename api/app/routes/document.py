from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List
from database.database import get_session
from database.models import ScrapedDocument, ScrapedDocumentRead

router = APIRouter(prefix="/documents", tags=["documents"])

@router.get("/{domain_id}", response_model=List[ScrapedDocumentRead])
def get_documents(domain_id: int, session: Session = Depends(get_session)):
    """Get all documents for a specific domain"""
    documents = session.exec(
        select(ScrapedDocument).where(ScrapedDocument.domain_id == domain_id)
    ).all()
    return documents 