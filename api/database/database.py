from sqlmodel import SQLModel, Session, create_engine
import os
from contextlib import contextmanager
from urllib.parse import quote_plus

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "Srnnkp@4999"),
    "database": os.getenv("DB_NAME", "mirialdbdev")
}

# URL encode the password to handle special characters
encoded_password = quote_plus(DB_CONFIG['password'])

DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{encoded_password}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

engine = create_engine(DATABASE_URL, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

@contextmanager
def get_db():
    """Context manager for database sessions"""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close() 