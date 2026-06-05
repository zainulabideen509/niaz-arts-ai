"""
Niaz Arts AI Backend - FastAPI Server
With MongoDB logging and analytics.
"""

import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from services.clip_service import CLIPService
from services.color_service import ColorService
from services.recommendation_service import RecommendationService
from services.product_sync import ProductSyncService
from services.mongo_service import MongoService
from services.notification_service import router as notify_router
from services.reviews_service import router as reviews_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("niaz-arts-ai")

app = FastAPI(
    title="Niaz Arts AI API",
    description="AI-powered painting recommendation system with analytics",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(notify_router)
app.include_router(reviews_router)

clip_service: Optional[CLIPService] = None
color_service: Optional[ColorService] = None
recommendation_service: Optional[RecommendationService] = None
mongo_service: Optional[MongoService] = None


@app.on_event("startup")
async def startup():
    global clip_service, color_service, recommendation_service, mongo_service

    logger.info("Loading CLIP model... (this takes 30-60s on first run)")
    clip_service = CLIPService()

    logger.info("Initializing color service...")
    color_service = ColorService()

    logger.info("Initializing recommendation service...")
    recommendation_service = RecommendationService(clip_service, color_service)

    logger.info("Connecting to MongoDB...")
    mongo_service = MongoService()

    if not recommendation_service.has_products():
        logger.warning("No products indexed! Run: python -m scripts.sync_from_csv")
    else:
        count = recommendation_service.product_count()
        logger.info(f"{count} products loaded and ready for recommendations.")


# ── Request/Response Models ──────────────────────────────────

class TextRecommendationRequest(BaseModel):
    query: str
    top_k: int = 6
    platform: str = "web"


class RecommendationResult(BaseModel):
    product_id: str
    shopify_id: str
    title: str
    image_url: str
    price: float
    score: float
    match_reason: str


class RecommendationResponse(BaseModel):
    success: bool
    recommendations: list[RecommendationResult]
    wall_colors: list[str] = []
    message: str = ""


# ── Recommendation Endpoints ─────────────────────────────────

@app.get("/")
async def health():
    has_products = recommendation_service.has_products() if recommendation_service else False
    mongo_connected = mongo_service.is_connected() if mongo_service else False
    return {
        "status": "ok",
        "service": "Niaz Arts AI API v2.0",
        "products_indexed": has_products,
        "mongodb_connected": mongo_connected,
    }


@app.post("/api/v1/image-recommendation", response_model=RecommendationResponse)
async def image_recommendation(
    image: UploadFile = File(...),
    top_k: int = Form(default=6),
    platform: str = Form(default="web"),
):
    if not recommendation_service or not recommendation_service.has_products():
        raise HTTPException(status_code=503, detail="Product index not ready.")

    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, and WebP accepted.")

    try:
        image_bytes = await image.read()
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large. Maximum 10MB.")

        results, wall_colors = recommendation_service.recommend_by_image(
            image_bytes, top_k=top_k
        )

        # Log to MongoDB
        if mongo_service and mongo_service.is_connected():
            mongo_service.log_image_recommendation(wall_colors, results, platform)

        return RecommendationResponse(
            success=True,
            recommendations=results,
            wall_colors=wall_colors,
            message=f"Found {len(results)} paintings matching your space.",
        )
    except Exception as e:
        logger.error(f"Image recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/text-recommendation", response_model=RecommendationResponse)
async def text_recommendation(request: TextRecommendationRequest):
    if not recommendation_service or not recommendation_service.has_products():
        raise HTTPException(status_code=503, detail="Product index not ready.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results = recommendation_service.recommend_by_text(
            request.query, top_k=request.top_k
        )

        # Log to MongoDB
        if mongo_service and mongo_service.is_connected():
            mongo_service.log_text_recommendation(request.query, results, request.platform)

        return RecommendationResponse(
            success=True,
            recommendations=results,
            message=f"Found {len(results)} paintings matching '{request.query}'.",
        )
    except Exception as e:
        logger.error(f"Text recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Analytics Endpoints ──────────────────────────────────────

@app.get("/api/v1/analytics/overview")
async def analytics_overview():
    """Get overall recommendation statistics."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    return {
        "totals": mongo_service.get_total_recommendations(days=30),
        "all_time": mongo_service.get_total_recommendations(days=3650),
    }


@app.get("/api/v1/analytics/top-queries")
async def analytics_top_queries():
    """Get most popular search queries."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    return {"top_queries": mongo_service.get_top_search_queries(limit=15)}


@app.get("/api/v1/analytics/popular-colors")
async def analytics_popular_colors():
    """Get most common wall colors from image uploads."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    return {"popular_colors": mongo_service.get_popular_wall_colors(limit=15)}


@app.get("/api/v1/analytics/top-paintings")
async def analytics_top_paintings():
    """Get most frequently recommended paintings."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    return {"top_paintings": mongo_service.get_most_recommended_paintings(limit=15)}


@app.get("/api/v1/analytics/platform-stats")
async def analytics_platform_stats():
    """Get usage by platform (web vs app)."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    return {"platforms": mongo_service.get_platform_stats()}


@app.get("/api/v1/analytics/hourly-usage")
async def analytics_hourly_usage():
    """Get recommendation counts by hour of day."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    return {"hourly": mongo_service.get_hourly_usage()}


@app.get("/api/v1/analytics/daily-usage")
async def analytics_daily_usage():
    """Get recommendation counts per day (last 30 days)."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    return {"daily": mongo_service.get_daily_usage(days=30)}


@app.get("/api/v1/analytics/recent-logs")
async def analytics_recent_logs():
    """Get the 20 most recent recommendation logs."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    return {"logs": mongo_service.get_recent_logs(limit=20)}


@app.get("/api/v1/analytics/full-dashboard")
async def analytics_full_dashboard():
    """Get all analytics data in one call (for admin dashboard)."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")

    return {
        "totals": mongo_service.get_total_recommendations(days=30),
        "all_time": mongo_service.get_total_recommendations(days=3650),
        "top_queries": mongo_service.get_top_search_queries(limit=10),
        "popular_colors": mongo_service.get_popular_wall_colors(limit=10),
        "top_paintings": mongo_service.get_most_recommended_paintings(limit=10),
        "platforms": mongo_service.get_platform_stats(),
        "hourly_usage": mongo_service.get_hourly_usage(),
        "daily_usage": mongo_service.get_daily_usage(days=30),
        "feedback_stats": mongo_service.get_feedback_stats(),
        "most_liked": mongo_service.get_most_liked_paintings(limit=10),
    }

# ── Feedback Endpoints ───────────────────────────────────────

class FeedbackRequest(BaseModel):
    painting_id: str
    action: str                                  # 'like' | 'dislike'
    painting_handle: Optional[str] = None
    recommendation_type: Optional[str] = None    # 'image' | 'text'
    user_query: Optional[str] = None
    timestamp: Optional[str] = None


@app.post("/api/v1/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Customer submits thumbs up/down on a recommended painting."""
    if request.action not in ["like", "dislike"]:
        raise HTTPException(status_code=400, detail="Action must be 'like' or 'dislike'.")

    if mongo_service and mongo_service.is_connected():
        mongo_service.save_feedback(
            product_id=request.painting_id,
            title=request.painting_handle or "",
            feedback_type=request.action,
            query=request.user_query or "",
            platform=request.recommendation_type or "app",
        )
        return {"status": "ok", "stored": True}
    else:
        raise HTTPException(status_code=503, detail="MongoDB not connected.")


@app.get("/api/v1/analytics/feedback-stats")
async def analytics_feedback_stats():
    """Get overall feedback statistics."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")
    return mongo_service.get_feedback_stats()


@app.get("/api/v1/analytics/most-liked")
async def analytics_most_liked():
    """Get most liked paintings from recommendations."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")
    return {"most_liked": mongo_service.get_most_liked_paintings(limit=15)}


@app.get("/api/v1/analytics/most-disliked")
async def analytics_most_disliked():
    """Get most disliked paintings from recommendations."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")
    return {"most_disliked": mongo_service.get_most_disliked_paintings(limit=15)}


@app.get("/api/v1/analytics/recent-feedback")
async def analytics_recent_feedback():
    """Get recent feedback entries."""
    if not mongo_service or not mongo_service.is_connected():
        raise HTTPException(status_code=503, detail="MongoDB not connected.")
    return {"feedback": mongo_service.get_recent_feedback(limit=20)}
@app.post("/api/v1/sync-products")
async def sync_products():
    try:
        sync_service = ProductSyncService(clip_service, color_service)
        count = await sync_service.sync()
        recommendation_service.reload_index()
        return {"success": True, "products_synced": count}
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)