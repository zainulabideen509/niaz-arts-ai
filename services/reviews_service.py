# ════════════════════════════════════════════════════════════════════════════
#  TRUSTOO (TrustWILL) REVIEWS PROXY  —  add to your FastAPI backend
#
#  WHY a proxy: the Trustoo API needs your PRIVATE token to sign every request
#  with HMAC-SHA256. That token must NEVER be shipped inside the app (anyone
#  could extract it from the APK). So the app calls YOUR backend, and the
#  backend signs + forwards the request to Trustoo. The private token stays
#  safe on Render as an environment variable.
#
#  SETUP:
#  1. Save this file as  services/reviews_service.py  in your backend.
#  2. In main.py add:
#         from services.reviews_service import router as reviews_router
#         app.include_router(reviews_router)
#  3. On Render, add two Environment Variables (from Trustoo admin):
#         TRUSTOO_PUBLIC_TOKEN   = <your public token>
#         TRUSTOO_PRIVATE_TOKEN  = <your private token>
#  4. requirements.txt already has httpx? if not, add:  httpx
# ════════════════════════════════════════════════════════════════════════════

import os
import time
import hmac
import hashlib
import httpx
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

TRUSTOO_BASE = "https://rapi.trustoo.io"
PUBLIC_TOKEN = os.environ.get("TRUSTOO_PUBLIC_TOKEN", "")
PRIVATE_TOKEN = os.environ.get("TRUSTOO_PRIVATE_TOKEN", "")


def _sign(params: dict) -> str:
    """HMAC-SHA256 over sorted key=value&... using the private token."""
    keys = sorted(params.keys())
    data_to_sign = "&".join(f"{k}={params[k]}" for k in keys)
    return hmac.new(
        PRIVATE_TOKEN.encode("utf-8"),
        data_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


@router.get("/api/v1/reviews")
async def get_reviews(
    product_id: str = Query(...),
    page: int = Query(1),
    page_size: int = Query(20),
):
    """Return Trustoo reviews for one product (proxied + signed)."""
    if not PUBLIC_TOKEN or not PRIVATE_TOKEN:
        raise HTTPException(status_code=503, detail="Trustoo tokens not configured.")

    timestamp = str(int(time.time()))
    params = {
        "product_ids": product_id,
        "page": str(page),
        "page_size": str(page_size),
        "timestamp": timestamp,
    }
    sign = _sign(params)

    headers = {
        "Public-Token": PUBLIC_TOKEN,
        "sign": sign,
        "timestamp": timestamp,
    }
    url = f"{TRUSTOO_BASE}/api/v1/openapi/get_reviews"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params, headers=headers)
            data = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Trustoo request failed: {e}")

    if data.get("code") != 0:
        raise HTTPException(status_code=502,
                            detail=f"Trustoo error: {data.get('message')}")

    d = data.get("data", {})
    reviews = d.get("list", [])
    page_info = d.get("page", {})

    # Trim to only what the app needs (smaller, faster payload)
    out = []
    ratings_sum = 0
    for rv in reviews:
        rating = rv.get("rating", 0)
        ratings_sum += rating
        out.append({
            "id": rv.get("id"),
            "author": rv.get("author") or "Anonymous",
            "rating": rating,
            "title": rv.get("title", ""),
            "content": rv.get("content", ""),
            "country": rv.get("author_country", ""),
            "date": rv.get("commented_at", ""),
            "verified": rv.get("is_verified", 0) == 1,
            "media": [m.get("url") for m in (rv.get("media") or []) if m.get("url")],
            "reply": (rv.get("reply") or {}).get("content") if rv.get("reply") else None,
        })

    return {
        "product_id": product_id,
        "count": page_info.get("count", len(out)),
        "page": page_info.get("page", page),
        "total_page": page_info.get("total_page", 1),
        "average": round(ratings_sum / len(out), 1) if out else 0,
        "reviews": out,
    }


# ── Create a review (POST) ───────────────────────────────────────────────────
# Signature for POST = HMAC-SHA256("timestamp=<ts>|<raw_json_body>", private)
import json as _json
from pydantic import BaseModel
from typing import Optional


class NewReview(BaseModel):
    product_id: str
    rating: int
    author: str
    content: str
    author_country: Optional[str] = "PK"
    title: Optional[str] = ""
    author_email: Optional[str] = ""


def _sign_post(timestamp: str, body: str) -> str:
    data_to_sign = f"timestamp={timestamp}|{body}"
    return hmac.new(
        PRIVATE_TOKEN.encode("utf-8"),
        data_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


@router.post("/api/v1/reviews/create")
async def create_review(review: NewReview):
    """Submit a new customer review to Trustoo for a product."""
    if not PUBLIC_TOKEN or not PRIVATE_TOKEN:
        raise HTTPException(status_code=503, detail="Trustoo tokens not configured.")
    if review.rating < 1 or review.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be 1-5.")
    if not review.content.strip():
        raise HTTPException(status_code=400, detail="Review content required.")

    timestamp = str(int(time.time()))
    payload = {
        "product_id": review.product_id,
        "rating": review.rating,
        "author": review.author.strip() or "Anonymous",
        "author_country": review.author_country or "PK",
        "content": review.content.strip(),
    }
    if review.title:
        payload["title"] = review.title.strip()
    if review.author_email:
        payload["author_email"] = review.author_email.strip()

    # IMPORTANT: sign the EXACT json string we send (no re-serialization)
    body_str = _json.dumps(payload, separators=(",", ":"))
    sign = _sign_post(timestamp, body_str)

    headers = {
        "Public-Token": PUBLIC_TOKEN,
        "Sign": sign,
        "Timestamp": timestamp,
        "Content-Type": "application/json",
    }
    url = f"{TRUSTOO_BASE}/api/v1/openapi/create_review"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(url, content=body_str, headers=headers)
            data = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Trustoo request failed: {e}")

    # Trustoo returns {"id": <new id>} on success, or an error code
    if isinstance(data, dict) and data.get("code") not in (None, 0):
        raise HTTPException(status_code=502,
                            detail=f"Trustoo error: {data.get('message')}")
    return {"success": True, "id": data.get("id") if isinstance(data, dict) else None}
