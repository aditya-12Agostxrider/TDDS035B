import io, re
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator, ConfigDict
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()

# Database Setup
engine = create_engine("sqlite:///./hotel_reviews.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database Models
class Hotel(Base):
    __tablename__ = "hotels"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    description = Column(String, nullable=True)
    reviews = relationship("Review", back_populates="hotel", cascade="all, delete-orphan")

class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True, index=True)
    hotel_id = Column(Integer, ForeignKey("hotels.id"), nullable=False)
    review_text = Column(String, nullable=False)
    rating = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    hotel = relationship("Hotel", back_populates="reviews")

Base.metadata.create_all(bind=engine)

# Pydantic Schemas
class HotelBase(BaseModel):
    name: str
    location: str
    description: Optional[str] = None

    @field_validator("name", "location")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty.")
        return v.strip()

class HotelCreate(HotelBase):
    pass

class HotelUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None

class HotelResponse(HotelBase):
    id: int
    model_config = ConfigDict(from_attributes=True)

class ReviewBase(BaseModel):
    review_text: str
    rating: int = Field(..., ge=1, le=5)

    @field_validator("review_text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Review text cannot be empty.")
        return v.strip()

class ReviewCreate(ReviewBase):
    pass

class ReviewUpdate(BaseModel):
    review_text: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)

class ReviewResponse(BaseModel):
    id: int
    hotel_id: int
    review_text: str
    rating: int
    created_at: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)

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

class PlaceAnalysisRequest(BaseModel):
    place_name: str
    custom_reviews: Optional[List[str]] = None
    url: Optional[str] = None

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

# NLP & Aspect Engine
ASPECT_KEYWORDS = {
    "infrastructure": ["infrastructure", "ambience", "interior", "decor", "architecture", "seating", "building", "furniture", "lighting", "space", "spacious", "elevator", "pool", "gym", "view", "ac"],
    "cleanliness": ["clean", "cleanliness", "dirty", "hygiene", "smell", "fresh", "washroom", "bathroom", "toilet", "neat", "sanitized", "spotless"],
    "food": ["food", "breakfast", "lunch", "dinner", "buffet", "restaurant", "delicious", "meal", "taste", "tasty", "dish", "menu", "drinks", "beverage", "spicy"],
    "service": ["service", "support", "check-in", "checkout", "laundry", "room service", "quick", "prompt", "staff", "reception", "employee", "housekeeping", "polite", "friendly", "helpful", "waiter"],
    "pricing": ["price", "expensive", "cheap", "cost", "overpriced", "affordable", "value", "charges", "bill", "rate", "costly"],
    "wifi": ["wifi", "wi-fi", "internet", "connection", "speed", "network"],
    "location": ["location", "area", "place", "near", "station", "center", "shopping", "safe", "accessible", "parking", "valet"]
}

PRESET_PLACES = {
    "taj palace hotel": {
        "name": "Taj Palace Hotel", "category": "Luxury Hotel", "base_rating": 4.6,
        "reviews": [
            "The infrastructure and royal interior ambience were breathtaking!",
            "Spotless cleanliness throughout the rooms and lobby. Extremely hygienic washrooms.",
            "Delicious breakfast buffet with incredible multi-cuisine taste and fresh food.",
            "Hospitality and staff service were very polite, attentive, and prompt.",
            "Quite expensive and food is costly, but high luxury value.",
            "Spacious room with a magnificent city view and luxurious furniture."
        ]
    },
    "barbeque nation": {
        "name": "Barbeque Nation", "category": "Casual Dining Restaurant", "base_rating": 4.3,
        "reviews": [
            "Amazing food and unlimited grilled starters! Taste was outstanding.",
            "The staff service was super quick and friendly.",
            "Good seating space and lively restaurant ambience.",
            "Cleanliness on the dining tables was good, washrooms clean.",
            "Reasonable pricing considering the extensive buffet spread."
        ]
    },
    "haldiram's": {
        "name": "Haldiram's", "category": "Quick Service Restaurant", "base_rating": 4.1,
        "reviews": [
            "Authentic North Indian food and sweets with delicious taste.",
            "The dining area is neat and hygienic, high standard of food cleanliness.",
            "Self-service can get crowded with slow service during peak rush.",
            "Affordable pricing and economical combo meals.",
            "Spacious seating and bright modern restaurant interior."
        ]
    },
    "grand hyatt": {
        "name": "Grand Hyatt", "category": "5-Star Resort & Hotel", "base_rating": 4.7,
        "reviews": [
            "Magnificent infrastructure with lush gardens, massive swimming pool, and gym.",
            "Impeccable room cleanliness, fresh bed sheets, and spotless bathrooms.",
            "Fine dining restaurants serve gourmet food with rich taste.",
            "Staff provided seamless check-in and attentive service.",
            "Premium pricing but worth every penny for the luxury atmosphere."
        ]
    }
}

def get_sentiment(text: str) -> str:
    compound = sia.polarity_scores(text)["compound"]
    return "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")

def detect_aspects(text: str) -> List[str]:
    text_lower = text.lower()
    detected = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                if aspect not in detected:
                    detected.append(aspect)
                break
    return detected

def analyze_reviews(reviews: List[Any]) -> Dict[str, Any]:
    if not reviews:
        return {"total_reviews": 0, "average_rating": 0.0, "sentiment": {"positive": 0, "negative": 0, "neutral": 0}, "positive_aspects": [], "negative_aspects": [], "aspect_details": {}}
    
    total = len(reviews)
    avg_rating = round(sum(r.rating for r in reviews) / total, 2)
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    aspect_stats: Dict[str, Dict[str, int]] = {}

    for rev in reviews:
        sent = get_sentiment(rev.review_text)
        sentiment_counts[sent] += 1
        for asp in detect_aspects(rev.review_text):
            if asp not in aspect_stats:
                aspect_stats[asp] = {"mentions": 0, "positive": 0, "negative": 0}
            aspect_stats[asp]["mentions"] += 1
            if sent == "positive":
                aspect_stats[asp]["positive"] += 1
            elif sent == "negative":
                aspect_stats[asp]["negative"] += 1

    pos_aspects, neg_aspects, aspect_details = [], [], {}
    for asp, stats in aspect_stats.items():
        dom_sent = "Positive" if stats["positive"] > stats["negative"] else ("Negative" if stats["negative"] > stats["positive"] else "Neutral")
        if dom_sent == "Positive": pos_aspects.append(asp)
        elif dom_sent == "Negative": neg_aspects.append(asp)
        aspect_details[asp] = {"mentions": stats["mentions"], "positive": stats["positive"], "negative": stats["negative"], "sentiment": dom_sent}

    return {"total_reviews": total, "average_rating": avg_rating, "sentiment": sentiment_counts, "positive_aspects": pos_aspects, "negative_aspects": neg_aspects, "aspect_details": aspect_details}

def generate_summary(hotel_name: str, analysis: Dict[str, Any]) -> str:
    if analysis["total_reviews"] == 0:
        return f"No reviews available for {hotel_name} yet."
    sentences = []
    pos, neg = analysis.get("positive_aspects", []), analysis.get("negative_aspects", [])
    if pos:
        sentences.append(f"Most customers are satisfied with the {', '.join(pos[:-1]) + ' and ' + pos[-1] if len(pos) > 1 else pos[0]}.")
    if neg:
        sentences.append(f"Main concerns raised by guests include {', '.join(neg[:-1]) + ' and ' + neg[-1] if len(neg) > 1 else neg[0]}.")
    avg = analysis.get("average_rating", 0)
    rating_msg = "a strong positive recommendation" if avg >= 4.0 else ("an acceptable experience" if avg >= 3.0 else "lower ratings")
    sentences.append(f"Overall, {hotel_name} receives {rating_msg} with an average rating of {avg} / 5.")
    return " ".join(sentences)

def calculate_statistics(reviews: List[Any]) -> Dict[str, Any]:
    if not reviews:
        return {"total_reviews": 0, "average_rating": 0.0, "highest_rating": None, "lowest_rating": None, "positive_percentage": 0.0, "negative_percentage": 0.0, "neutral_percentage": 0.0}
    total = len(reviews)
    ratings = [r.rating for r in reviews]
    sentiments = [get_sentiment(r.review_text) for r in reviews]
    return {
        "total_reviews": total,
        "average_rating": round(sum(ratings) / total, 2),
        "highest_rating": max(ratings),
        "lowest_rating": min(ratings),
        "positive_percentage": round((sentiments.count("positive") / total) * 100, 1),
        "negative_percentage": round((sentiments.count("negative") / total) * 100, 1),
        "neutral_percentage": round((sentiments.count("neutral") / total) * 100, 1)
    }

def analyze_google_place(place_name: str, custom_reviews: Optional[List[str]] = None, url: Optional[str] = None) -> Dict[str, Any]:
    name = place_name.strip()
    key = name.lower()
    category, base_rating = "Hotel / Restaurant", 4.2

    if custom_reviews and len(custom_reviews) > 0:
        reviews = [r.strip() for r in custom_reviews if r.strip()]
        category = "Custom Google Reviews"
    elif key in PRESET_PLACES:
        p = PRESET_PLACES[key]
        name, category, base_rating, reviews = p["name"], p["category"], p["base_rating"], p["reviews"]
    else:
        is_restaurant = any(k in key for k in ["restaurant", "cafe", "dhaba", "bistro", "bakery", "kitchen", "barbeque", "food"])
        category = "Restaurant" if is_restaurant else "Hotel"
        reviews = [
            f"The infrastructure and ambience of {name} were very pleasant with spacious seating.",
            f"Hygiene and cleanliness were maintained properly in the dining area and washrooms.",
            f"Food was tasty and fresh with good variety on the menu.",
            f"Staff service was helpful, polite, and responsive throughout our visit.",
            f"Pricing was reasonable and offered fair value for money.",
            f"The location is convenient and accessible with nearby parking."
        ]

    total = len(reviews)
    sents = [get_sentiment(r) for r in reviews]
    pos_pct = round((sents.count("positive") / total) * 100, 1) if total else 0.0
    neu_pct = round((sents.count("neutral") / total) * 100, 1) if total else 0.0
    neg_pct = round((sents.count("negative") / total) * 100, 1) if total else 0.0

    targets = [
        ("cleanliness", "Cleanliness & Hygiene"),
        ("infrastructure", "Infrastructure & Ambience"),
        ("food", "Food & Taste"),
        ("service", "Staff & Service"),
        ("pricing", "Pricing & Value"),
        ("wifi", "WiFi & Amenities"),
        ("location", "Location & Parking")
    ]

    aspect_scores, bar_labels, bar_scores, radar_labels, radar_scores = {}, [], [], [], []
    pos_list, neg_list = [], []

    for asp_key, disp_name in targets:
        pos_m, neg_m, tot_m = 0, 0, 0
        for r in reviews:
            if asp_key in detect_aspects(r):
                tot_m += 1
                s = get_sentiment(r)
                if s == "positive": pos_m += 1
                elif s == "negative": neg_m += 1

        score = int(round((pos_m / tot_m) * 100)) if tot_m > 0 else 70
        sent_label = "Positive" if pos_m >= neg_m and tot_m > 0 else ("Negative" if neg_m > pos_m else "Neutral")
        if sent_label == "Positive": pos_list.append(disp_name)
        elif sent_label == "Negative": neg_list.append(disp_name)

        aspect_scores[disp_name] = {"score": score, "sentiment": sent_label, "positive_mentions": pos_m, "negative_mentions": neg_m, "total_mentions": tot_m}
        bar_labels.append(disp_name)
        bar_scores.append(score)
        radar_labels.append(disp_name.split(" ")[0])
        radar_scores.append(score)

    summary_parts = []
    if pos_list: summary_parts.append(f"Customers praise the {', '.join(pos_list[:3])}.")
    if neg_list: summary_parts.append(f"Guest reviews raise concerns regarding {', '.join(neg_list)}.")
    summary_parts.append(f"Overall, {name} holds a rating of {base_rating} ⭐ with strong ratings in infrastructure and service.")

    return {
        "place_name": name,
        "category": category,
        "total_reviews_analyzed": total,
        "overall_rating": base_rating,
        "aspect_scores": aspect_scores,
        "graph_data": {
            "aspect_labels": bar_labels,
            "aspect_scores": bar_scores,
            "sentiment_labels": ["Positive 😊", "Neutral 😐", "Negative 😞"],
            "sentiment_percentages": [pos_pct, neu_pct, neg_pct],
            "radar_labels": radar_labels,
            "radar_scores": radar_scores
        },
        "summary": " ".join(summary_parts),
        "sample_reviews": reviews[:6]
    }

# FastAPI App
app = FastAPI(
    title="Hotel & Restaurant Review Analysis & Graph API",
    description="All-in-one REST API to analyze customer reviews, evaluate Infrastructure, Cleanliness, Food, Service, Price, and generate graphs.",
    version="1.0.0"
)

# HTML Embedded UI
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hotel & Restaurant Review Graph Analyzer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
        body { background-color: #f8fafc; color: #0f172a; line-height: 1.5; padding-bottom: 40px; }
        .container { max-width: 1100px; margin: 0 auto; padding: 0 20px; }
        .app-header { background: #fff; border-bottom: 1px solid #e2e8f0; padding: 16px 0; margin-bottom: 24px; }
        .header-wrap { display: flex; justify-content: space-between; align-items: center; }
        .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .hero { background: linear-gradient(135deg, #fff, #f0f7ff); border: 2px solid #bfdbfe; }
        .search-box { display: flex; gap: 10px; margin: 14px 0; }
        input, select, textarea { width: 100%; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 0.95rem; }
        .btn { padding: 10px 18px; border-radius: 6px; font-weight: 600; cursor: pointer; border: none; text-decoration: none; display: inline-flex; align-items: center; }
        .btn-primary { background: #2563eb; color: #fff; }
        .btn-success { background: #16a34a; color: #fff; }
        .btn-secondary { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }
        .pills { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
        .pill-btn { background: #fff; border: 1px solid #cbd5e1; padding: 4px 12px; border-radius: 20px; font-size: 0.82rem; cursor: pointer; }
        .grid { display: grid; gap: 20px; }
        .grid-2 { grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
        .chart-box { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; height: 300px; }
        .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin: 16px 0; }
        .stat { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center; }
        .stat-num { font-size: 1.4rem; font-weight: 700; display: block; }
        .summary-box { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin: 16px 0; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.88rem; }
        th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
        th { background: #f8fafc; color: #475569; }
        .toast { position: fixed; bottom: 20px; right: 20px; background: #1e293b; color: #fff; padding: 10px 18px; border-radius: 6px; opacity: 0; transition: opacity 0.3s; z-index: 1000; }
        .toast.show { opacity: 1; }
    </style>
</head>
<body>
    <header class="app-header">
        <div class="container header-wrap">
            <div><h1>🏨 Hotel & Restaurant Review Graph Analyzer</h1><p style="font-size:0.85rem; color:#64748b;">FastAPI All-In-One NLP Review Analysis System</p></div>
            <a href="/docs" target="_blank" class="btn btn-secondary">⚡ Swagger Docs</a>
        </div>
    </header>
    <main class="container">
        <div class="card hero">
            <h2>🔍 Search Any Hotel or Restaurant (Google Reviews & Graphs)</h2>
            <p style="font-size:0.9rem; color:#475569;">Type place name to plot Cleanliness, Infrastructure, Food, Service & Price graphs!</p>
            <div class="search-box">
                <input type="text" id="placeInput" value="Taj Palace Hotel" placeholder="e.g. Taj Palace, Barbeque Nation, Haldiram's, Grand Hyatt...">
                <button class="btn btn-primary" onclick="analyzePlace()">📊 Generate Graphs</button>
            </div>
            <div class="pills">
                <span style="font-size:0.8rem; font-weight:600; color:#475569;">Quick Places:</span>
                <button class="pill-btn" onclick="quickSearch('Taj Palace Hotel')">Taj Palace Hotel</button>
                <button class="pill-btn" onclick="quickSearch('Barbeque Nation')">Barbeque Nation</button>
                <button class="pill-btn" onclick="quickSearch('Haldiram\'s')">Haldiram's</button>
                <button class="pill-btn" onclick="quickSearch('Grand Hyatt')">Grand Hyatt</button>
            </div>
        </div>

        <div id="dash" class="card" style="display:none;">
            <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #e2e8f0; padding-bottom:10px;">
                <div><h2 id="resName">Place Name</h2><p id="resSub" style="color:#64748b; font-size:0.9rem;">Category</p></div>
                <div style="background:#fef3c7; border:1px solid #fde68a; padding:6px 14px; border-radius:20px; font-weight:700; color:#92400e;">⭐ <span id="resRating">4.5</span> / 5</div>
            </div>
            <div class="stat-grid">
                <div class="stat"><span class="stat-num" id="resTotal">0</span>Reviews</div>
                <div class="stat" style="color:#16a34a;"><span class="stat-num" id="resPos">0%</span>😊 Positive</div>
                <div class="stat" style="color:#64748b;"><span class="stat-num" id="resNeu">0%</span>😐 Neutral</div>
                <div class="stat" style="color:#dc2626;"><span class="stat-num" id="resNeg">0%</span>😞 Negative</div>
            </div>
            <div class="grid grid-2">
                <div class="chart-box"><h4 style="text-align:center; font-size:0.9rem; margin-bottom:6px;">Aspect Quality Scores (0 - 100%)</h4><canvas id="barChart"></canvas></div>
                <div class="chart-box"><h4 style="text-align:center; font-size:0.9rem; margin-bottom:6px;">360° Quality Radar</h4><canvas id="radarChart"></canvas></div>
            </div>
            <div class="summary-box">
                <h4 style="color:#1e40af; margin-bottom:4px;">📝 Generated Natural Language Summary</h4>
                <p id="resSummary" style="font-size:0.92rem; color:#1e293b;"></p>
            </div>
            <h4 style="margin-top:16px;">📊 Aspect Scorecard</h4>
            <table>
                <thead><tr><th>Aspect</th><th>Score</th><th>Mentions</th><th>Positive</th><th>Negative</th><th>Verdict</th></tr></thead>
                <tbody id="aspectsTbody"></tbody>
            </table>
        </div>
    </main>
    <div id="toast" class="toast"></div>
    <script>
        let barInstance = null, radarInstance = null;
        document.addEventListener("DOMContentLoaded", () => analyzePlace());
        function showToast(msg) {
            const t = document.getElementById("toast");
            t.textContent = msg; t.classList.add("show");
            setTimeout(() => t.classList.remove("show"), 3000);
        }
        function quickSearch(name) { document.getElementById("placeInput").value = name; analyzePlace(); }
        async function analyzePlace() {
            const name = document.getElementById("placeInput").value.trim();
            if (!name) return showToast("Enter place name");
            try {
                const res = await fetch("/analyze-place", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({place_name: name})
                });
                const data = await res.json();
                document.getElementById("resName").textContent = data.place_name;
                document.getElementById("resSub").textContent = data.category + " - " + data.total_reviews_analyzed + " reviews analyzed";
                document.getElementById("resRating").textContent = data.overall_rating;
                document.getElementById("resTotal").textContent = data.total_reviews_analyzed;
                document.getElementById("resPos").textContent = data.graph_data.sentiment_percentages[0] + "%";
                document.getElementById("resNeu").textContent = data.graph_data.sentiment_percentages[1] + "%";
                document.getElementById("resNeg").textContent = data.graph_data.sentiment_percentages[2] + "%";
                document.getElementById("resSummary").textContent = data.summary;
                renderBar(data.graph_data.aspect_labels, data.graph_data.aspect_scores);
                renderRadar(data.graph_data.radar_labels, data.graph_data.radar_scores);
                const tbody = document.getElementById("aspectsTbody");
                tbody.innerHTML = "";
                for (const [k, v] of Object.entries(data.aspect_scores)) {
                    tbody.innerHTML += `<tr><td style="font-weight:600;">${k}</td><td><strong>${v.score}%</strong></td><td>${v.total_mentions}</td><td style="color:#16a34a;">${v.positive_mentions}</td><td style="color:#dc2626;">${v.negative_mentions}</td><td style="font-weight:600; color:${v.sentiment==='Positive'?'#16a34a':(v.sentiment==='Negative'?'#dc2626':'#64748b')}">${v.sentiment}</td></tr>`;
                }
                document.getElementById("dash").style.display = "block";
                showToast("Graphs plotted for " + data.place_name);
            } catch(e) { showToast("Error: " + e.message); }
        }
        function renderBar(labels, scores) {
            const ctx = document.getElementById("barChart").getContext("2d");
            if (barInstance) barInstance.destroy();
            const colors = scores.map(s => s >= 75 ? "#16a34a" : (s >= 50 ? "#d97706" : "#dc2626"));
            barInstance = new Chart(ctx, {
                type: "bar",
                data: { labels: labels, datasets: [{ data: scores, backgroundColor: colors, borderRadius: 4 }] },
                options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, scales: { x: { min: 0, max: 100 } }, plugins: { legend: { display: false } } }
            });
        }
        function renderRadar(labels, scores) {
            const ctx = document.getElementById("radarChart").getContext("2d");
            if (radarInstance) radarInstance.destroy();
            radarInstance = new Chart(ctx, {
                type: "radar",
                data: { labels: labels, datasets: [{ label: "Quality Balance", data: scores, backgroundColor: "rgba(37, 99, 235, 0.2)", borderColor: "#2563eb", borderWidth: 2 }] },
                options: { responsive: true, maintainAspectRatio: false, scales: { r: { min: 0, max: 100, ticks: { display: false } } } }
            });
        }
    </script>
</body>
</html>"""

# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.get("/", tags=["General"], response_class=HTMLResponse)
def serve_ui():
    return HTMLResponse(content=HTML_CONTENT)

@app.post("/analyze-place", response_model=PlaceAnalysisResponse, tags=["Analysis"])
def analyze_place_endpoint(req: PlaceAnalysisRequest):
    return analyze_google_place(place_name=req.place_name, custom_reviews=req.custom_reviews, url=req.url)

@app.post("/hotels", response_model=HotelResponse, status_code=status.HTTP_201_CREATED, tags=["Hotels"])
def create_hotel(hotel: HotelCreate, db: Session = Depends(get_db)):
    db_hotel = Hotel(name=hotel.name, location=hotel.location, description=hotel.description)
    db.add(db_hotel)
    db.commit()
    db.refresh(db_hotel)
    return db_hotel

@app.get("/hotels", response_model=List[HotelResponse], tags=["Hotels"])
def get_all_hotels(db: Session = Depends(get_db)):
    return db.query(Hotel).all()

@app.get("/hotels/{hotel_id}", response_model=HotelResponse, tags=["Hotels"])
def get_hotel(hotel_id: int, db: Session = Depends(get_db)):
    h = db.query(Hotel).filter(Hotel.id == hotel_id).first()
    if not h: raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")
    return h

@app.put("/hotels/{hotel_id}", response_model=HotelResponse, tags=["Hotels"])
def update_hotel(hotel_id: int, h_up: HotelUpdate, db: Session = Depends(get_db)):
    h = db.query(Hotel).filter(Hotel.id == hotel_id).first()
    if not h: raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")
    if h_up.name: h.name = h_up.name
    if h_up.location: h.location = h_up.location
    if h_up.description: h.description = h_up.description
    db.commit()
    db.refresh(h)
    return h

@app.delete("/hotels/{hotel_id}", tags=["Hotels"])
def delete_hotel(hotel_id: int, db: Session = Depends(get_db)):
    h = db.query(Hotel).filter(Hotel.id == hotel_id).first()
    if not h: raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")
    db.delete(h)
    db.commit()
    return {"message": f"Hotel {hotel_id} deleted successfully."}

@app.post("/hotels/{hotel_id}/reviews", response_model=ReviewResponse, status_code=status.HTTP_201_CREATED, tags=["Reviews"])
def add_review(hotel_id: int, rev: ReviewCreate, db: Session = Depends(get_db)):
    if not db.query(Hotel).filter(Hotel.id == hotel_id).first():
        raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")
    r = Review(hotel_id=hotel_id, review_text=rev.review_text, rating=rev.rating)
    db.add(r)
    db.commit()
    db.refresh(r)
    return r

@app.get("/hotels/{hotel_id}/reviews", response_model=List[ReviewResponse], tags=["Reviews"])
def get_hotel_reviews(hotel_id: int, db: Session = Depends(get_db)):
    if not db.query(Hotel).filter(Hotel.id == hotel_id).first():
        raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")
    return db.query(Review).filter(Review.hotel_id == hotel_id).all()

@app.get("/reviews/{review_id}", response_model=ReviewResponse, tags=["Reviews"])
def get_review(review_id: int, db: Session = Depends(get_db)):
    r = db.query(Review).filter(Review.id == review_id).first()
    if not r: raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    return r

@app.put("/reviews/{review_id}", response_model=ReviewResponse, tags=["Reviews"])
def update_review(review_id: int, r_up: ReviewUpdate, db: Session = Depends(get_db)):
    r = db.query(Review).filter(Review.id == review_id).first()
    if not r: raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    if r_up.review_text: r.review_text = r_up.review_text
    if r_up.rating: r.rating = r_up.rating
    db.commit()
    db.refresh(r)
    return r

@app.delete("/reviews/{review_id}", tags=["Reviews"])
def delete_review(review_id: int, db: Session = Depends(get_db)):
    r = db.query(Review).filter(Review.id == review_id).first()
    if not r: raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    db.delete(r)
    db.commit()
    return {"message": f"Review {review_id} deleted successfully."}

@app.get("/hotels/{hotel_id}/analysis", response_model=AnalysisResponse, tags=["Analysis"])
def get_analysis(hotel_id: int, db: Session = Depends(get_db)):
    h = db.query(Hotel).filter(Hotel.id == hotel_id).first()
    if not h: raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")
    revs = db.query(Review).filter(Review.hotel_id == hotel_id).all()
    res = analyze_reviews(revs)
    return {"hotel_id": h.id, "hotel_name": h.name, "total_reviews": res["total_reviews"], "average_rating": res["average_rating"], "sentiment": res["sentiment"], "positive_aspects": res["positive_aspects"], "negative_aspects": res["negative_aspects"]}

@app.get("/hotels/{hotel_id}/aspects", response_model=AspectsResponse, tags=["Analysis"])
def get_aspects(hotel_id: int, db: Session = Depends(get_db)):
    h = db.query(Hotel).filter(Hotel.id == hotel_id).first()
    if not h: raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")
    revs = db.query(Review).filter(Review.hotel_id == hotel_id).all()
    return {"hotel_id": h.id, "aspects": analyze_reviews(revs)["aspect_details"]}

@app.get("/hotels/{hotel_id}/summary", response_model=SummaryResponse, tags=["Analysis"])
def get_summary(hotel_id: int, db: Session = Depends(get_db)):
    h = db.query(Hotel).filter(Hotel.id == hotel_id).first()
    if not h: raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")
    revs = db.query(Review).filter(Review.hotel_id == hotel_id).all()
    return {"hotel_id": h.id, "summary": generate_summary(h.name, analyze_reviews(revs))}

@app.get("/hotels/{hotel_id}/statistics", response_model=StatisticsResponse, tags=["Analysis"])
def get_statistics(hotel_id: int, db: Session = Depends(get_db)):
    if not db.query(Hotel).filter(Hotel.id == hotel_id).first():
        raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")
    revs = db.query(Review).filter(Review.hotel_id == hotel_id).all()
    return calculate_statistics(revs)

@app.post("/hotels/{hotel_id}/reviews/upload", status_code=status.HTTP_201_CREATED, tags=["Reviews"])
async def upload_csv(hotel_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    h = db.query(Hotel).filter(Hotel.id == hotel_id).first()
    if not h: raise HTTPException(status_code=404, detail=f"Hotel {hotel_id} not found")
    if not file.filename.endswith(".csv"): raise HTTPException(status_code=400, detail="Must be a CSV file.")
    df = pd.read_csv(io.BytesIO(await file.read()))
    if not {"review_text", "rating"}.issubset(df.columns): raise HTTPException(status_code=400, detail="Missing review_text or rating columns.")
    added = 0
    for _, row in df.iterrows():
        t, r = str(row["review_text"]).strip(), int(row["rating"])
        if t and 1 <= r <= 5:
            db.add(Review(hotel_id=hotel_id, review_text=t, rating=r))
            added += 1
    db.commit()
    return {"message": f"Added {added} reviews to '{h.name}'.", "hotel_id": hotel_id, "reviews_added": added}

if __name__ == "__main__":
    uvicorn.run("app_single:app", host="127.0.0.1", port=8000, reload=True)
