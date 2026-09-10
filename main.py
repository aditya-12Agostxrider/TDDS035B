"""
Hotel Customer Review Analysis and Summarization API
Main application file with FastAPI endpoints, NLP Analysis, Statistics, Summaries, CSV Upload, Web UI, and Google Place Graph Analyzer.
"""

import io
from pathlib import Path
from typing import List
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

import models
import schemas
import analysis
from database import engine, get_db

# Automatically create database tables if they do not already exist
models.Base.metadata.create_all(bind=engine)

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Hotel & Restaurant Review Analysis & Graph Summarization API",
    description="A comprehensive REST API to analyze customer reviews, evaluate Infrastructure, Cleanliness, Food, Service, and Pricing, and plot visual review graphs.",
    version="1.1.0"
)

# Mount static files directory for CSS and JS
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# ==========================================
# WEB UI & ROOT ENDPOINT
# ==========================================

@app.get("/", tags=["General"], summary="Frontend Web Dashboard")
def serve_frontend():
    """
    Serves the interactive HTML frontend dashboard.
    """
    index_file = BASE_DIR / "templates" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Welcome to the Hotel Customer Review Analysis API!",
        "docs_url": "/docs",
        "status": "active"
    }


# ==========================================
# GOOGLE PLACE & RESTAURANT REVIEW GRAPH API
# ==========================================

@app.post(
    "/analyze-place",
    response_model=schemas.PlaceAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Google Place & Restaurant Graph Analysis"],
    summary="Analyze any Hotel or Restaurant on Google and Generate Graph Metrics"
)
def analyze_place_reviews(request: schemas.PlaceAnalysisRequest):
    """
    Analyze customer reviews for ANY Hotel or Restaurant on Google:
    - Calculates scores (0-100%) for **Cleanliness, Infrastructure, Food, Service, and Pricing**.
    - Prepares numerical graph data for **Aspect Bar Charts, 360° Radar Charts, and Sentiment Doughnut Charts**.
    - Generates a natural language executive summary.
    """
    result = analysis.analyze_google_place(
        place_name=request.place_name,
        custom_reviews=request.custom_reviews,
        url=request.url
    )
    return result


# ==========================================
# HOTEL CRUD ENDPOINTS
# ==========================================

@app.post(
    "/hotels",
    response_model=schemas.HotelResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Hotels"],
    summary="Create a new hotel"
)
def create_hotel(hotel: schemas.HotelCreate, db: Session = Depends(get_db)):
    """
    Create a new hotel in the database.
    - **name**: Name of the hotel (Required)
    - **location**: City / area of the hotel (Required)
    - **description**: Optional hotel description
    """
    db_hotel = models.Hotel(
        name=hotel.name,
        location=hotel.location,
        description=hotel.description
    )
    db.add(db_hotel)
    db.commit()
    db.refresh(db_hotel)
    return db_hotel


@app.get(
    "/hotels",
    response_model=List[schemas.HotelResponse],
    status_code=status.HTTP_200_OK,
    tags=["Hotels"],
    summary="Get all hotels"
)
def get_all_hotels(db: Session = Depends(get_db)):
    """
    Retrieve a list of all registered hotels.
    """
    hotels = db.query(models.Hotel).all()
    return hotels


@app.get(
    "/hotels/{hotel_id}",
    response_model=schemas.HotelResponse,
    status_code=status.HTTP_200_OK,
    tags=["Hotels"],
    summary="Get a single hotel by ID"
)
def get_hotel(hotel_id: int, db: Session = Depends(get_db)):
    """
    Retrieve details of a specific hotel by its ID.
    Returns 404 if the hotel does not exist.
    """
    db_hotel = db.query(models.Hotel).filter(models.Hotel.id == hotel_id).first()
    if not db_hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with id {hotel_id} not found"
        )
    return db_hotel


@app.put(
    "/hotels/{hotel_id}",
    response_model=schemas.HotelResponse,
    status_code=status.HTTP_200_OK,
    tags=["Hotels"],
    summary="Update a hotel by ID"
)
def update_hotel(hotel_id: int, hotel_update: schemas.HotelUpdate, db: Session = Depends(get_db)):
    """
    Update information for an existing hotel.
    Allows partial updates of name, location, and description.
    """
    db_hotel = db.query(models.Hotel).filter(models.Hotel.id == hotel_id).first()
    if not db_hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with id {hotel_id} not found"
        )

    if hotel_update.name is not None:
        db_hotel.name = hotel_update.name
    if hotel_update.location is not None:
        db_hotel.location = hotel_update.location
    if hotel_update.description is not None:
        db_hotel.description = hotel_update.description

    db.commit()
    db.refresh(db_hotel)
    return db_hotel


@app.delete(
    "/hotels/{hotel_id}",
    status_code=status.HTTP_200_OK,
    tags=["Hotels"],
    summary="Delete a hotel by ID"
)
def delete_hotel(hotel_id: int, db: Session = Depends(get_db)):
    """
    Delete a hotel and all its associated reviews from the database.
    """
    db_hotel = db.query(models.Hotel).filter(models.Hotel.id == hotel_id).first()
    if not db_hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with id {hotel_id} not found"
        )

    db.delete(db_hotel)
    db.commit()
    return {"message": f"Hotel with id {hotel_id} and its associated reviews were deleted successfully."}


# ==========================================
# REVIEW CRUD ENDPOINTS
# ==========================================

@app.post(
    "/hotels/{hotel_id}/reviews",
    response_model=schemas.ReviewResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Reviews"],
    summary="Add a new review for a hotel"
)
def add_review(hotel_id: int, review: schemas.ReviewCreate, db: Session = Depends(get_db)):
    """
    Add a customer review for a specified hotel.
    - **review_text**: Review comment from the customer (Required)
    - **rating**: Rating integer between 1 and 5 (Required)
    """
    db_hotel = db.query(models.Hotel).filter(models.Hotel.id == hotel_id).first()
    if not db_hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with id {hotel_id} not found"
        )

    db_review = models.Review(
        hotel_id=hotel_id,
        review_text=review.review_text,
        rating=review.rating
    )
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return db_review


@app.get(
    "/hotels/{hotel_id}/reviews",
    response_model=List[schemas.ReviewResponse],
    status_code=status.HTTP_200_OK,
    tags=["Reviews"],
    summary="Get all reviews for a hotel"
)
def get_hotel_reviews(hotel_id: int, db: Session = Depends(get_db)):
    """
    Retrieve all reviews submitted for a specific hotel.
    """
    db_hotel = db.query(models.Hotel).filter(models.Hotel.id == hotel_id).first()
    if not db_hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with id {hotel_id} not found"
        )

    reviews = db.query(models.Review).filter(models.Review.hotel_id == hotel_id).all()
    return reviews


@app.get(
    "/reviews/{review_id}",
    response_model=schemas.ReviewResponse,
    status_code=status.HTTP_200_OK,
    tags=["Reviews"],
    summary="Get a single review by ID"
)
def get_review(review_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a single review by its review ID.
    """
    db_review = db.query(models.Review).filter(models.Review.id == review_id).first()
    if not db_review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review with id {review_id} not found"
        )
    return db_review


@app.put(
    "/reviews/{review_id}",
    response_model=schemas.ReviewResponse,
    status_code=status.HTTP_200_OK,
    tags=["Reviews"],
    summary="Update a review by ID"
)
def update_review(review_id: int, review_update: schemas.ReviewUpdate, db: Session = Depends(get_db)):
    """
    Update review text and/or rating for an existing review.
    """
    db_review = db.query(models.Review).filter(models.Review.id == review_id).first()
    if not db_review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review with id {review_id} not found"
        )

    if review_update.review_text is not None:
        db_review.review_text = review_update.review_text
    if review_update.rating is not None:
        db_review.rating = review_update.rating

    db.commit()
    db.refresh(db_review)
    return db_review


@app.delete(
    "/reviews/{review_id}",
    status_code=status.HTTP_200_OK,
    tags=["Reviews"],
    summary="Delete a review by ID"
)
def delete_review(review_id: int, db: Session = Depends(get_db)):
    """
    Delete a single review by its review ID.
    """
    db_review = db.query(models.Review).filter(models.Review.id == review_id).first()
    if not db_review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review with id {review_id} not found"
        )

    db.delete(db_review)
    db.commit()
    return {"message": f"Review with id {review_id} deleted successfully."}


# ==========================================
# REVIEW ANALYSIS & SUMMARIZATION ENDPOINTS
# ==========================================

@app.get(
    "/hotels/{hotel_id}/analysis",
    response_model=schemas.AnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Analyze hotel reviews (Sentiment & Aspects)"
)
def get_hotel_analysis(hotel_id: int, db: Session = Depends(get_db)):
    """
    Main analysis endpoint: Analyzes all customer reviews for a specified hotel.
    Returns total count, average rating, sentiment breakdown, and positive/negative aspects.
    """
    db_hotel = db.query(models.Hotel).filter(models.Hotel.id == hotel_id).first()
    if not db_hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with id {hotel_id} not found"
        )

    reviews = db.query(models.Review).filter(models.Review.hotel_id == hotel_id).all()
    analysis_result = analysis.analyze_reviews(reviews)

    return {
        "hotel_id": db_hotel.id,
        "hotel_name": db_hotel.name,
        "total_reviews": analysis_result["total_reviews"],
        "average_rating": analysis_result["average_rating"],
        "sentiment": analysis_result["sentiment"],
        "positive_aspects": analysis_result["positive_aspects"],
        "negative_aspects": analysis_result["negative_aspects"]
    }


@app.get(
    "/hotels/{hotel_id}/aspects",
    response_model=schemas.AspectsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Get detailed aspect-based analysis for a hotel"
)
def get_hotel_aspects(hotel_id: int, db: Session = Depends(get_db)):
    """
    Aspect-level breakdown: Returns mention counts and sentiment breakdown for each aspect (room, wifi, staff, etc.).
    """
    db_hotel = db.query(models.Hotel).filter(models.Hotel.id == hotel_id).first()
    if not db_hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with id {hotel_id} not found"
        )

    reviews = db.query(models.Review).filter(models.Review.hotel_id == hotel_id).all()
    analysis_result = analysis.analyze_reviews(reviews)

    return {
        "hotel_id": db_hotel.id,
        "aspects": analysis_result["aspect_details"]
    }


@app.get(
    "/hotels/{hotel_id}/summary",
    response_model=schemas.SummaryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Get generated customer-friendly hotel summary"
)
def get_hotel_summary(hotel_id: int, db: Session = Depends(get_db)):
    """
    Generates a concise, natural language summary of what guests like and dislike about the hotel.
    """
    db_hotel = db.query(models.Hotel).filter(models.Hotel.id == hotel_id).first()
    if not db_hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with id {hotel_id} not found"
        )

    reviews = db.query(models.Review).filter(models.Review.hotel_id == hotel_id).all()
    analysis_result = analysis.analyze_reviews(reviews)
    summary_text = analysis.generate_summary(db_hotel.name, analysis_result)

    return {
        "hotel_id": db_hotel.id,
        "summary": summary_text
    }


@app.get(
    "/hotels/{hotel_id}/statistics",
    response_model=schemas.StatisticsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Get statistical review metrics for a hotel"
)
def get_hotel_statistics(hotel_id: int, db: Session = Depends(get_db)):
    """
    Calculates detailed numerical metrics: average, highest, lowest rating, and sentiment percentages.
    """
    db_hotel = db.query(models.Hotel).filter(models.Hotel.id == hotel_id).first()
    if not db_hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with id {hotel_id} not found"
        )

    reviews = db.query(models.Review).filter(models.Review.hotel_id == hotel_id).all()
    stats = analysis.calculate_statistics(reviews)
    return stats


# ==========================================
# CSV UPLOAD ENDPOINT
# ==========================================

@app.post(
    "/hotels/{hotel_id}/reviews/upload",
    status_code=status.HTTP_201_CREATED,
    tags=["Reviews"],
    summary="Upload reviews in bulk via CSV file"
)
async def upload_reviews_csv(hotel_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a CSV file containing reviews for a hotel.
    Expected CSV columns: **review_text**, **rating**
    """
    db_hotel = db.query(models.Hotel).filter(models.Hotel.id == hotel_id).first()
    if not db_hotel:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Hotel with id {hotel_id} not found"
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a CSV (.csv)."
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading CSV file: {str(e)}"
        )

    required_columns = {"review_text", "rating"}
    if not required_columns.issubset(df.columns):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV file must contain 'review_text' and 'rating' columns. Found columns: {list(df.columns)}"
        )

    added_count = 0
    for _, row in df.iterrows():
        text = str(row["review_text"]).strip()
        try:
            rating_val = int(row["rating"])
        except (ValueError, TypeError):
            continue

        if text and 1 <= rating_val <= 5:
            new_review = models.Review(
                hotel_id=hotel_id,
                review_text=text,
                rating=rating_val
            )
            db.add(new_review)
            added_count += 1

    db.commit()
    return {
        "message": f"Successfully uploaded and added {added_count} reviews to hotel '{db_hotel.name}'.",
        "hotel_id": hotel_id,
        "reviews_added": added_count
    }
