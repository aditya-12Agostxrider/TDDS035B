"""
SQLAlchemy ORM models defining the database schema for Hotels and Reviews.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class Hotel(Base):
    """
    Hotel Table: Represents hotels in the system.
    """
    __tablename__ = "hotels"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    description = Column(String, nullable=True)

    # Relationship to reviews: When a hotel is deleted, all its reviews are also deleted (cascade)
    reviews = relationship("Review", back_populates="hotel", cascade="all, delete-orphan")


class Review(Base):
    """
    Review Table: Represents customer reviews belonging to a hotel.
    """
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)
    hotel_id = Column(Integer, ForeignKey("hotels.id"), nullable=False)
    review_text = Column(String, nullable=False)
    rating = Column(Integer, nullable=False)  # Scale 1 to 5
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back to hotel
    hotel = relationship("Hotel", back_populates="reviews")
