from typing import Optional, List
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship

class DomainBase(SQLModel):
    domain_name: str = Field(max_length=255)
    domain_url: str = Field(max_length=255, unique=True)

class Domain(DomainBase, table=True):
    __tablename__ = "domain"
    
    domain_id: Optional[int] = Field(default=None, primary_key=True)
    documents: List["ScrapedDocument"] = Relationship(back_populates="domain")

class DomainRead(DomainBase):
    domain_id: int

class DomainCreate(DomainBase):
    pass

class ScrapedDocumentBase(SQLModel):
    url: str = Field(max_length=255)
    url_hash: str = Field(max_length=32)
    content_raw: str
    last_scraped: datetime = Field(default_factory=datetime.utcnow)

class ScrapedDocument(ScrapedDocumentBase, table=True):
    __tablename__ = "scraped_document"
    
    document_id: Optional[int] = Field(default=None, primary_key=True)
    domain_id: int = Field(foreign_key="domain.domain_id")
    domain: Domain = Relationship(back_populates="documents")

class ScrapedDocumentRead(ScrapedDocumentBase):
    document_id: int
    domain_id: int