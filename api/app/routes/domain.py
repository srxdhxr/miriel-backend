from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session
from database.database import get_session
from database.models import Domain, DomainCreate, DomainRead

router = APIRouter(prefix="/domains", tags=["domains"])

@router.post("/", response_model=DomainRead)
def create_domain(domain: DomainCreate, session: Session = Depends(get_session)):
    new_domain = Domain.from_orm(domain)
    session.add(new_domain)
    session.commit()
    session.refresh(new_domain)
    return new_domain 