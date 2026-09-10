"""
Database connection and session configuration using SQLAlchemy with SQLite.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite database file path (saved in the project folder as hotel_reviews.db)
SQLALCHEMY_DATABASE_URL = "sqlite:///./hotel_reviews.db"

# create_engine establishes connection to the SQLite database.
# check_same_thread: False allows FastAPI to interact with SQLite across multiple threads safely.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# SessionLocal is the database session factory.
# Each API request will get its own database session.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for SQLAlchemy database models.
Base = declarative_base()


def get_db():
    """
    Dependency function that provides a database session to API route handlers.
    Ensures the database session is always closed after request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
