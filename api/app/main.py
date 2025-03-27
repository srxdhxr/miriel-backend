from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.database import create_db_and_tables
from app.routes import domain_settings, domain, recommendations, scraper, vectorizer, document, health

app = FastAPI(
    title="Mirial API",
    description="Backend API for Mirial chatbot",
    version="1.0.0"
)

# Define allowed origins
origins = [
    "http://localhost:5173", 
    "http://localhost:5174",   # Vite default dev server
    "http://localhost:3000",    # React default
    "http://localhost:8005",    # Your current frontend
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8005",
]

# Add CORS middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Replace ["*"] with specific origins
    allow_credentials=True,     # Allow cookies
    allow_methods=["*"],        # Allow all methods
    allow_headers=["*"],        # Allow all headers
    expose_headers=["*"]        # Expose all headers
)



# Include routers
app.include_router(domain.router)
app.include_router(domain_settings.router)
app.include_router(recommendations.router)
app.include_router(document.router)
app.include_router(scraper.router)
app.include_router(vectorizer.router)
app.include_router(health.router)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()