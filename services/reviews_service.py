# ════════════════════════════════════════════════════════════════════════════
#  TRUSTOO (TrustWILL) REVIEWS PROXY  —  v2 (robust + clear errors)
#  Save as services/reviews_service.py in your FastAPI backend.
#
#  Render env vars required:
#     TRUSTOO_PUBLIC_TOKEN   = <your public token>
#     TRUSTOO_PRIVATE_TOKEN  = <your private token>   (only needed for writing)
# ════════════════════════════════════════════════════════════════════════════

import os
import time
import hmac
import json as _json
import hashlib
import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

TRUSTOO_BASE = "https://rapi.trustoo.io"
PUBLIC_TOKEN = os.environ.get("TRUSTOO_PUBLIC_TOKEN", "")
PRIVATE_TOKEN = os.environ.get("TRUSTOO_PRIVATE_TOKEN", "")


@router.get("/api/v1/reviews")
async def get_reviews(
    product_id: str = Query(...),
    page: int = Query(1),
    page_size: int = Query(20),
):
    """Return Trustoo reviews for one product. GET uses Public-Token only."""
    if not PUBLIC_TOKEN:
        raise HTTPException(status_code=503, detail="TRUSTOO_PUBLIC_TOKEN not set.")

    params = {
        "product_ids": product_id,
        "page": str(page),
        "page_size": str(page_size),
    }
    headers = {"Public-Token": PUBLIC_TOKEN}
    url = f"{TRUSTOO_BASE}/api/v1/openapi/get_reviews"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params, headers=headers)
            raw = r.text
            try:
                data = r.json()
            except Exception:
                # Surface the real upstream message instead of a generic error
                raise HTTPException(status_code=502,
                                    detail=f"Trustoo non-JSON ({r.status_code}): {raw[:300]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Trustoo request error: {e}")

    # If Trustoo returns an error code, pass its message through so we can see it
    if isinstance(data, dict) and data.get("code") not in (None, 0):
        raise HTTPException(status_code=502,
                            detail=f"Trustoo code {data.get('code')}: {data.get('message')}")

    d = data.get("data", {}) if isinstance(data, dict) else {}
    reviews = d.get("list", []) or []
    page_info = d.get("page", {}) or {}

    out = []
    ratings_sum = 0
    for rv in reviews:
        rating = rv.get("rating", 0) or 0
        ratings_sum += rating
        out.append({
            "id": rv.get("id"),
            "author": rv.get("author") or "Anonymous",
            "rating": rating,
            "title": rv.get("title", "") or "",
            "content": rv.get("content", "") or "",
            "country": rv.get("author_country", "") or "",
            "date": rv.get("commented_at", "") or "",
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


# ── Create a review (POST) — needs the private token to sign ─────────────────
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
    return hmac.new(PRIVATE_TOKEN.encode("utf-8"),
                    data_to_sign.encode("utf-8"),
                    hashlib.sha256).hexdigest()


@router.post("/api/v1/reviews/create")
async def create_review(review: NewReview):
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

    if isinstance(data, dict) and data.get("code") not in (None, 0):
        raise HTTPException(status_code=502,
                            detail=f"Trustoo error: {data.get('message')}")
    return {"success": True, "id": data.get("id") if isinstance(data, dict) else None}


# ── Debug endpoint: shows raw Trustoo response so we can see the real issue ──
@router.get("/api/v1/reviews/debug")
async def reviews_debug(product_id: str = Query("0")):
    info = {
        "public_token_set": bool(PUBLIC_TOKEN),
        "public_token_preview": (PUBLIC_TOKEN[:6] + "...") if PUBLIC_TOKEN else None,
        "private_token_set": bool(PRIVATE_TOKEN),
    }
    if not PUBLIC_TOKEN:
        info["error"] = "TRUSTOO_PUBLIC_TOKEN missing on server"
        return info
    url = f"{TRUSTOO_BASE}/api/v1/openapi/get_reviews"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url,
                                  params={"product_ids": product_id, "page": "1", "page_size": "5"},
                                  headers={"Public-Token": PUBLIC_TOKEN})
            info["status_code"] = r.status_code
            info["raw_response"] = r.text[:800]
    except Exception as e:
        info["request_error"] = str(e)
    return info
