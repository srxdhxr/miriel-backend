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
    settings: List["DomainSettings"] = Relationship(back_populates="domain")
    recommendations: List["Recommendation"] = Relationship(back_populates="domain")

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

class RecommendationBase(SQLModel):
    title: str = Field(max_length=255)
    description: str = Field(max_length=1000)
    link: str = Field(max_length=500)
    domain_id: int = Field(foreign_key="domain.domain_id")

class Recommendation(RecommendationBase, table=True):
    __tablename__ = "recommendation"
    
    recommendation_id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_deleted: bool = Field(default=False)
    domain: Domain = Relationship(back_populates="recommendations")

class RecommendationRead(RecommendationBase):
    recommendation_id: int
    created_at: datetime
    updated_at: datetime

class RecommendationCreate(RecommendationBase):
    pass

class RecommendationUpdate(SQLModel):
    title: Optional[str] = None
    description: Optional[str] = None
    link: Optional[str] = None

class DomainSettingsBase(SQLModel):
    chatbot_title: str = Field(max_length=255)
    initial_text: str
    description: str = Field(default="", max_length=500)
    

    font_family: str = Field(default="Inter, system-ui, sans-serif")
    
    # Chat specific colors
    chat_header_container_color: str = Field(default="#FFFFFF")
    chat_title_color: str = Field(default="#6D28D9")
    description_color: str = Field(default="#6D28D9")
    message_box_color: str = Field(default="#F9FAFB")
    user_box_color: str = Field(default="#6D28D9")
    assistant_box_color: str = Field(default="#FFFFFF")
    user_text_color: str = Field(default="#FFFFFF")
    assistant_text_color: str = Field(default="#1F2937")
    source_container_color: str = Field(default="#F9FAFB")
    source_title_color: str = Field(default="#1F2937")
    source_text_color: str = Field(default="#6D28D9")
    input_container_color: str = Field(default="#F9FAFB")
    input_box_color: str = Field(default="#F9FAFB")
    rec_container_color: str = Field(default="#F9FAFB")
    rec_title_color: str = Field(default="#1F2937")
    rec_tile_color: str = Field(default="#F9FAFB")
    rec_tile_text_color: str = Field(default="#1F2937")
    show_recommendations: bool = Field(default=True)
    added_at: datetime = Field(default_factory=datetime.utcnow)
    removed_at: Optional[datetime] = Field(default=None)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class DomainSettings(DomainSettingsBase, table=True):
    __tablename__ = "domain_settings"
    
    setting_id: Optional[int] = Field(default=None, primary_key=True)
    domain_id: int = Field(foreign_key="domain.domain_id")
    domain: Domain = Relationship(back_populates="settings")

class DomainSettingsRead(DomainSettingsBase):
    setting_id: int
    domain_id: int

class DomainSettingsCreate(SQLModel):
    domain_id: int
    chatbot_title: str = Field(max_length=255)
    initial_text: str
    description: str = Field(default="", max_length=500)
    font_family: str = Field(default="Inter, system-ui, sans-serif")
    chat_header_container_color: str = Field(default="#FFFFFF")
    chat_title_color: str = Field(default="#6D28D9")
    description_color: str = Field(default="#6D28D9")
    message_box_color: str = Field(default="#F9FAFB")
    user_box_color: str = Field(default="#6D28D9")
    assistant_box_color: str = Field(default="#FFFFFF")
    user_text_color: str = Field(default="#FFFFFF")
    assistant_text_color: str = Field(default="#1F2937")
    source_container_color: str = Field(default="#F9FAFB")
    source_title_color: str = Field(default="#1F2937")
    source_text_color: str = Field(default="#6D28D9")
    input_container_color: str = Field(default="#F9FAFB")
    input_box_color: str = Field(default="#F9FAFB")
    rec_container_color: str = Field(default="#F9FAFB")
    rec_title_color: str = Field(default="#1F2937")
    rec_tile_color: str = Field(default="#F9FAFB")
    rec_tile_text_color: str = Field(default="#1F2937")
    show_recommendations: bool = Field(default=True)

class DomainSettingsUpdate(SQLModel):
    chatbot_title: Optional[str] = None
    initial_text: Optional[str] = None
    description: Optional[str] = None