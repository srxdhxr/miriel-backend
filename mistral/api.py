from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import base64
import tarfile
from io import BytesIO
import os
from typing import Optional, List, Dict
from .fast_rag import FastRAG
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = None

class VectorUpload(BaseModel):
    vector_data: str
    domain_id: Optional[int] = None

class QueryInput(BaseModel):
    query: str
    domain_id: Optional[int] = 1
    temperature: Optional[float] = 0.7

class Source(BaseModel):
    url: str
    last_scraped: str
    domain_id: int

class QueryResponse(BaseModel):
    response: Optional[str] = None
    error: Optional[str] = None
    sources: List[Source]

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup if vector store exists"""
    try:
        global rag_system
        base_path = os.getenv("VECTOR_STORE_PATH", "/app")
        if os.path.exists(os.path.join(base_path, "vectorstore")):
            rag_system = FastRAG(base_path=base_path)
            await rag_system.initialize()
            
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    global rag_system
    if rag_system:
        await rag_system.cleanup()

@app.get("/")
async def root():
    """Test endpoint"""
    return {"status": "ok", "message": "API is running"}

@app.post("/upload_vectors")
async def upload_vectors(vector_data: VectorUpload):
    """Receive and save vector store data"""
    try:
        base_path = os.getenv("VECTOR_STORE_PATH", "/app")
        
        # Decode base64 data
        decoded_data = base64.b64decode(vector_data.vector_data)
        
        # Create a BytesIO object for the tar data
        tar_buffer = BytesIO(decoded_data)
        
        # Extract to vector store directory
        vector_store_path = os.path.join(base_path, "vectorstore")
        os.makedirs(vector_store_path, exist_ok=True)
        
        with tarfile.open(fileobj=tar_buffer, mode='r:gz') as tar:
            tar.extractall(path=vector_store_path)
        
        # Cleanup old RAG system if exists
        global rag_system
        if rag_system:
            await rag_system.cleanup()
        
        # Reinitialize RAG system with new vectors
        rag_system = FastRAG(
            domain_id=vector_data.domain_id or 1,
            base_path=base_path
        )
        await rag_system.initialize()
            
        return {"message": "Vector store updated successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(input_data: QueryInput):
    """Query using RAG system"""
    try:
        global rag_system
        if rag_system is None:
            try:
                base_path = os.getenv("VECTOR_STORE_PATH", "/app")
                rag_system = FastRAG(
                    domain_id=input_data.domain_id,
                    base_path=base_path
                )
                await rag_system.initialize()
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=f"Vector store not found: {str(e)}")
            
        result = await rag_system.get_response(
            query=input_data.query,
            temperature=input_data.temperature
        )
        
        return QueryResponse(
            response=result["response"],
            error=result.get("error"),
            sources=result["sources"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 