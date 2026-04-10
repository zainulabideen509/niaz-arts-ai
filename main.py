"""
Niaz Arts AI Backend — FastAPI Server
Exposes endpoints for image-based and text-based painting recommendations.
"""

import os
import io
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from services.clip_service import CLIPService
from services.color_service import ColorService
from services.recommendation_service import RecommendationService
from services.product_sync import ProductSyncService

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("niaz-arts-ai")

# ── FastAPI App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Niaz Arts AI API",
    description="AI-powered painting recommendation system for Niaz Arts",
    version="1.0.0",
)

# Allow requests from your Shopify store, Flutter app, and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://niazarts.pk",
        "https://www.niazarts.pk",
        "https://niaz-arts.myshopify.com",  # update with your actual .myshopify.com domain
        "http://localhost:*",
        "*",  # Remove this in production — here for development convenience
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize Services (loaded once at startup) ─────────────────────
clip_service: Optional[CLIPService] = None
color_service: Optional[ColorService] = None
recommendation_service: Optional[RecommendationService] = None


@app.on_event("startup")
async def startup():
    global clip_service, color_service, recommendation_service

    logger.info("Loading CLIP model... (this takes 30-60s on first run)")
    clip_service = CLIPService()

    logger.info("Initializing color service...")
    color_service = ColorService()

    logger.info("Initializing recommendation service...")
    recommendation_service = RecommendationService(clip_service, color_service)

    # Check if product index exists, if not, prompt to sync
    if not recommendation_service.has_products():
        logger.warning(
            "⚠️  No products indexed! Run: python -m scripts.sync_products"
        )
    else:
        count = recommendation_service.product_count()
        logger.info(f"✅ {count} products loaded and ready for recommendations.")


# ── Request/Response Models ──────────────────────────────────────────
class TextRecommendationRequest(BaseModel):
    query: str
    top_k: int = 6


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
    wall_colors: list[str] = []  # hex colors extracted from wall (for image recs)
    message: str = ""


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/")
async def health():
    """Health check endpoint."""
    has_products = recommendation_service.has_products() if recommendation_service else False
    return {
        "status": "ok",
        "service": "Niaz Arts AI API",
        "products_indexed": has_products,
    }


@app.post("/api/v1/image-recommendation", response_model=RecommendationResponse)
async def image_recommendation(
    image: UploadFile = File(...),
    top_k: int = Form(default=6),
):
    """
    Upload a photo of a wall/room. Returns paintings that match the
    color palette, style, and aesthetic of the space.
    """
    if not recommendation_service or not recommendation_service.has_products():
        raise HTTPException(status_code=503, detail="Product index not ready. Run sync first.")

    # Validate file type
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, and WebP images are accepted.")

    try:
        image_bytes = await image.read()
        if len(image_bytes) > 10 * 1024 * 1024:  # 10 MB limit
            raise HTTPException(status_code=400, detail="Image too large. Maximum 10MB.")

        results, wall_colors = recommendation_service.recommend_by_image(
            image_bytes, top_k=top_k
        )

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
    """
    Describe what you're looking for in natural language.
    E.g. "calm blue abstract painting for bedroom"
    """
    if not recommendation_service or not recommendation_service.has_products():
        raise HTTPException(status_code=503, detail="Product index not ready. Run sync first.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results = recommendation_service.recommend_by_text(
            request.query, top_k=request.top_k
        )

        return RecommendationResponse(
            success=True,
            recommendations=results,
            message=f"Found {len(results)} paintings matching '{request.query}'.",
        )
    except Exception as e:
        logger.error(f"Text recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/sync-products")
async def sync_products():
    """
    Manually trigger a product sync from Shopify.
    In production, call this via a Shopify webhook on product create/update/delete.
    """
    try:
        sync_service = ProductSyncService(clip_service, color_service)
        count = await sync_service.sync()
        # Reload the recommendation service index
        recommendation_service.reload_index()
        return {"success": True, "products_synced": count}
    except Exception as e:
        logger.error(f"Sync failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
