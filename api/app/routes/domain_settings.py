from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from datetime import datetime
from typing import List

from database.database import get_session
from database.models import (
    DomainSettings,
    DomainSettingsCreate,
    DomainSettingsRead,
    DomainSettingsUpdate,
    Domain
)

router = APIRouter(prefix="/domain-settings", tags=["domain-settings"])

@router.get("/domain/{domain_id}", response_model=DomainSettingsRead)
def get_domain_settings(domain_id: int, session: Session = Depends(get_session)):
    # Check if domain exists
    domain = session.get(Domain, domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    # Get the latest settings for the domain based on setting_id
    latest_settings = session.exec(
        select(DomainSettings)
        .where(DomainSettings.domain_id == domain_id)
        .where(DomainSettings.removed_at == None)  # Only get active settings
        .order_by(DomainSettings.setting_id.desc())  # Order by setting_id descending
        .limit(1)  # Get only the latest one
    ).first()
    
    if not latest_settings:
        raise HTTPException(status_code=404, detail="No active settings found for this domain")
    
    return latest_settings

@router.post("/", response_model=DomainSettingsRead)
def create_domain_settings(settings: DomainSettingsCreate, session: Session = Depends(get_session)):
    # Check if domain exists
    domain = session.get(Domain, settings.domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    # Deactivate any existing active settings for this domain
    existing_settings = session.exec(
        select(DomainSettings)
        .where(DomainSettings.domain_id == settings.domain_id)
        .where(DomainSettings.removed_at == None)
    ).all()
    
    for existing in existing_settings:
        existing.removed_at = datetime.utcnow()
    
    # Create new settings with provided values
    new_settings = DomainSettings(
        domain_id=settings.domain_id,
        **settings.dict(exclude={'domain_id'})
    )
    
    session.add(new_settings)
    session.commit()
    session.refresh(new_settings)
    return new_settings

@router.put("/{setting_id}", response_model=DomainSettingsRead)
def update_domain_settings(
    setting_id: int,
    settings_update: DomainSettingsUpdate,
    session: Session = Depends(get_session)
):
    db_settings = session.get(DomainSettings, setting_id)
    if not db_settings:
        raise HTTPException(status_code=404, detail="Settings not found")
    
    if db_settings.removed_at:
        raise HTTPException(status_code=400, detail="Cannot update removed settings")
    
    # Update only provided fields
    settings_data = settings_update.dict(exclude_unset=True)
    for key, value in settings_data.items():
        setattr(db_settings, key, value)
    
    db_settings.updated_at = datetime.utcnow()
    session.commit()
    session.refresh(db_settings)
    return db_settings

@router.delete("/{setting_id}")
def remove_domain_settings(setting_id: int, session: Session = Depends(get_session)):
    db_settings = session.get(DomainSettings, setting_id)
    if not db_settings:
        raise HTTPException(status_code=404, detail="Settings not found")
    
    if db_settings.removed_at:
        raise HTTPException(status_code=400, detail="Settings already removed")
    
    db_settings.removed_at = datetime.utcnow()
    session.commit()
    return {"message": "Settings removed successfully"} 