from fastapi import APIRouter, Depends, HTTPException, Body
from sqlmodel import Session, select, create_engine
from typing import List, Optional
from datetime import datetime
from database.database import get_session
from database.models import Domain, Recommendation, RecommendationCreate, RecommendationRead, RecommendationUpdate
from pydantic import BaseModel

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Define the model for delete operation
class RecommendationDelete(BaseModel):
    recommendation_ids: List[int]

# 1. Create table API endpoint
@router.post("/create-table")
def create_recommendations_table(session: Session = Depends(get_session)):
    """Create the recommendations table if it doesn't exist"""
    try:
        # Get the engine from the session
        engine = session.get_bind()
        # Create the table
        Recommendation.metadata.create_all(engine)
        return {"message": "Recommendations table created successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create table: {str(e)}")

# 2. Get recommendations by IDs and domain_id
@router.get("/domain/{domain_id}", response_model=List[RecommendationRead])
def get_domain_recommendations(
    domain_id: int, 
    recommendation_ids: Optional[List[int]] = None,
    session: Session = Depends(get_session)
):
    """Get recommendations for a domain, optionally filtered by recommendation IDs"""
    # Check if domain exists
    domain = session.get(Domain, domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    # Build the query
    query = select(Recommendation).where(
        Recommendation.domain_id == domain_id,
        Recommendation.is_deleted == False
    )
    
    # Add recommendation_id filter if provided
    if recommendation_ids:
        query = query.where(Recommendation.recommendation_id.in_(recommendation_ids))
    
    # Execute the query
    recommendations = session.exec(query).all()
    return recommendations

# 3. Add a new recommendation
@router.post("/", response_model=RecommendationRead)
def create_recommendation(
    recommendation: RecommendationCreate,
    session: Session = Depends(get_session)
):
    """Create a new recommendation"""
    # Check if domain exists
    domain = session.get(Domain, recommendation.domain_id)
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")
    
    # Create new recommendation
    db_recommendation = Recommendation.from_orm(recommendation)
    
    session.add(db_recommendation)
    session.commit()
    session.refresh(db_recommendation)
    
    return db_recommendation

# 4. Delete recommendations by IDs
@router.post("/delete", status_code=200)
def delete_recommendations(
    delete_data: RecommendationDelete,
    hard_delete: bool = False,  # Add query parameter for hard delete
    session: Session = Depends(get_session)
):
    """Delete multiple recommendations by IDs
    
    Args:
        delete_data: RecommendationDelete object containing IDs to delete
        hard_delete: If True, performs permanent deletion. If False, soft deletes (default)
    """
    if not delete_data.recommendation_ids:
        raise HTTPException(status_code=400, detail="No recommendation IDs provided")
    
    # Get recommendations to delete
    recommendations = session.exec(
        select(Recommendation)
        .where(
            Recommendation.recommendation_id.in_(delete_data.recommendation_ids),
            Recommendation.is_deleted == False
        )
    ).all()
    
    if not recommendations:
        raise HTTPException(status_code=404, detail="No matching recommendations found")
    
    deleted_ids = [r.recommendation_id for r in recommendations]
    
    if hard_delete:
        # Permanent deletion
        for recommendation in recommendations:
            session.delete(recommendation)
        message = f"Successfully permanently deleted {len(recommendations)} recommendations"
    else:
        # Soft deletion
        for recommendation in recommendations:
            recommendation.is_deleted = True
            recommendation.updated_at = datetime.utcnow()
        message = f"Successfully soft deleted {len(recommendations)} recommendations"
    
    session.commit()
    
    return {
        "message": message,
        "deleted_ids": deleted_ids,
        "deletion_type": "hard" if hard_delete else "soft"
    }

# 5. Edit a recommendation by ID
@router.put("/{recommendation_id}", response_model=RecommendationRead)
def update_recommendation(
    recommendation_id: int,
    recommendation_update: RecommendationUpdate,
    session: Session = Depends(get_session)
):
    """Update a recommendation by ID"""
    # Get the recommendation
    db_recommendation = session.get(Recommendation, recommendation_id)
    if not db_recommendation:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    
    if db_recommendation.is_deleted:
        raise HTTPException(status_code=400, detail="Cannot update deleted recommendation")
    
    # Update only provided fields
    update_data = recommendation_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_recommendation, key, value)
    
    # Update the updated_at timestamp
    db_recommendation.updated_at = datetime.utcnow()
    
    session.commit()
    session.refresh(db_recommendation)
    
    return db_recommendation 