"""
Pydantic schemas for data validation and API serialization.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


# ==========================================
# HOTEL SCHEMAS
# ==========================================

class HotelBase(BaseModel):
    name: str = Field(..., description="Hotel name (cannot be empty)")
    location: str = Field(..., description="City or area where the hotel is located")
    description: Optional[str] = Field(None, description="Brief description of the hotel")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Hotel name cannot be empty or whitespace.")
        return v.strip()

    @field_validator("location")
    @classmethod
    def location_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Hotel location cannot be empty or whitespace.")
        return v.strip()


class HotelCreate(HotelBase):
    pass


class HotelUpdate(BaseModel):
    name: Optional[str] = Field(None, description="Updated hotel name")
    location: Optional[str] = Field(None, description="Updated location")
    description: Optional[str] = Field(None, description="Updated description")

    @field_validator("name", "location")
    @classmethod
    def fields_must_not_be_empty_if_provided(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Field cannot be empty or whitespace.")
        return v.strip() if v is not None else None


class HotelResponse(HotelBase):
    id: int

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# REVIEW SCHEMAS
# ==========================================

class ReviewBase(BaseModel):
    review_text: str = Field(..., description="Customer review text")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (lowest) to 5 (highest)")

    @field_validator("review_text")
    @classmethod
    def review_text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Review text cannot be empty or whitespace.")
        return v.strip()


class ReviewCreate(ReviewBase):
    pass


class ReviewUpdate(BaseModel):
    review_text: Optional[str] = Field(None, description="Updated review text")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Updated rating from 1 to 5")

    @field_validator("review_text")
    @classmethod
    def text_must_not_be_empty_if_provided(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("Review text cannot be empty or whitespace.")
        return v.strip() if v is not None else None


class ReviewResponse(BaseModel):
    id: int
    hotel_id: int
    review_text: str
    rating: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# ANALYSIS & SUMMARY SCHEMAS
# ==========================================

class SentimentCounts(BaseModel):
    positive: int
    negative: int
    neutral: int


class AnalysisResponse(BaseModel):
    hotel_id: int
    hotel_name: str
    total_reviews: int
    average_rating: float
    sentiment: SentimentCounts
    positive_aspects: List[str]
    negative_aspects: List[str]


class AspectDetail(BaseModel):
    mentions: int
    positive: int
    negative: int
    sentiment: str


class AspectsResponse(BaseModel):
    hotel_id: int
    aspects: Dict[str, AspectDetail]


class SummaryResponse(BaseModel):
    hotel_id: int
    summary: str


class StatisticsResponse(BaseModel):
    total_reviews: int
    average_rating: float
    highest_rating: Optional[int] = None
    lowest_rating: Optional[int] = None
    positive_percentage: float
    negative_percentage: float
    neutral_percentage: float


# ==========================================
# GOOGLE PLACE & GRAPH SCHEMAS
# ==========================================

class PlaceAnalysisRequest(BaseModel):
    place_name: str = Field(..., description="Hotel or Restaurant name to search on Google")
    custom_reviews: Optional[List[str]] = Field(None, description="Optional custom pasted reviews")
    url: Optional[str] = Field(None, description="Optional Google Maps / Hotel URL")

    @field_validator("place_name")
    @classmethod
    def place_name_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Place name cannot be empty.")
        return v.strip()


class AspectScoreDetail(BaseModel):
    score: int
    sentiment: str
    positive_mentions: int
    negative_mentions: int
    total_mentions: int


class GraphData(BaseModel):
    aspect_labels: List[str]
    aspect_scores: List[int]
    sentiment_labels: List[str]
    sentiment_percentages: List[float]
    radar_labels: List[str]
    radar_scores: List[int]


class PlaceAnalysisResponse(BaseModel):
    place_name: str
    category: str
    total_reviews_analyzed: int
    overall_rating: float
    aspect_scores: Dict[str, AspectScoreDetail]
    graph_data: GraphData
    summary: str
    sample_reviews: List[str]
