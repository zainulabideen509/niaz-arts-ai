# ════════════════════════════════════════════════════════════════════════════
#  TRUSTOO (TrustWILL) REVIEWS PROXY — FIX FOR RENDER.COM
# ════════════════════════════════════════════════════════════════════════════
import os
import time
import hmac
import hashlib
import json  # Fixed the missing json import issue!
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

TRUSTOO_BASE = "https://rapi.trustoo.io"
PUBLIC_TOKEN = os.environ.get("TRUSTOO_PUBLIC_TOKEN", "")
PRIVATE_TOKEN = os.environ.get("TRUSTOO_PRIVATE_TOKEN", "")

# ─── MODELS ─────────────────────────────────────────────────────────
class ReviewCreate(BaseModel):
    product_id: str
    rating: int
    author: str
    content: str
    title: Optional[str] = ""
    author_email: Optional[str] = ""
    author_country: Optional[str] = "PK"

# ─── HELPER: GENERATE HMAC-SHA256 SIGNATURE ─────────────────────────
def generate_trustoo_signature(query_params: dict, body_str: str, timestamp: str) -> str:
    """
    Trustoo updated their security to HMAC-SHA256.
    1. Gather all params including timestamp.
    2. Sort alphabetically.
    3. Append body with '|' if exists.
    4. Sign with PRIVATE_TOKEN.
    """
    params = query_params.copy()
    params['timestamp'] = timestamp
    
    # Sort alphabetically
    sorted_keys = sorted(params.keys())
    query_string = "&".join(f"{k}={params[k]}" for k in sorted_keys)
    
    data_to_sign = query_string
    if body_str:
        data_to_sign += "|" + body_str
        
    signature = hmac.new(
        PRIVATE_TOKEN.encode('utf-8'),
        data_to_sign.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return signature.lower()


# ─── GET REVIEWS LIST ───────────────────────────────────────────────
@router.get("/api/v1/reviews")
async def get_reviews(product_id: str, page: int = 1, page_size: int = 20):
    timestamp = str(int(time.time()))
    
    # 1. Fetch the reviews list
    query_params = {
        "product_id": str(product_id),
        "page": str(page),
        "page_size": str(page_size)
    }
    
    sign = generate_trustoo_signature(query_params, "", timestamp)
    
    headers = {
        "Public-Token": PUBLIC_TOKEN,
        "Timestamp": timestamp,
        "Sign": sign
    }
    
    url = f"{TRUSTOO_BASE}/api/v1/openapi/get_review_list"
    
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, params=query_params, headers=headers)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail="Trustoo GET failed")
        data = r.json()

    if data.get("code") != 0:
        raise HTTPException(status_code=400, detail=data.get("message", "Error fetching reviews"))
        
    trustoo_data = data.get("data", {})
    page_info = trustoo_data.get("page", {})
    trustoo_list = trustoo_data.get("list", [])
    
    # 2. Fetch the average rating (Trustoo stores this in a separate endpoint)
    rating_query = {"product_id": str(product_id)}
    rating_sign = generate_trustoo_signature(rating_query, "", timestamp)
    rating_headers = {
        "Public-Token": PUBLIC_TOKEN,
        "Timestamp": timestamp,
        "Sign": rating_sign
    }
    rating_url = f"{TRUSTOO_BASE}/api/v1/openapi/get_rating"
    
    average_rating = 5.0 # Fallback
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            rr = await client.get(rating_url, params=rating_query, headers=rating_headers)
            if rr.status_code == 200 and rr.json().get("code") == 0:
                average_rating = float(rr.json().get("data", {}).get("rating_value", 5.0))
    except Exception:
        pass # If rating fails, just continue with fallback
    
    # 3. Map Trustoo response to EXACTLY what Flutter expects
    formatted_reviews = []
    for item in trustoo_list:
        # Handle different image formats returned by Trustoo
        media_list = []
        if isinstance(item.get("images"), list):
            for m in item["images"]:
                if isinstance(m, dict) and "url" in m:
                    media_list.append(m["url"])
                elif isinstance(m, str):
                    media_list.append(m)

        formatted_reviews.append({
            "id": item.get("id", item.get("review_id", "")),
            "author": item.get("author", "Anonymous"),
            "rating": item.get("rating", 5),
            "title": item.get("title", ""),
            "content": item.get("content", ""),
            "country": item.get("author_country", ""),
            "date": item.get("created_at", ""),
            "verified": True, 
            "media": media_list,
            "reply": item.get("reply_content", "")
        })
        
    return {
        "count": page_info.get("count", len(formatted_reviews)),
        "average": average_rating,
        "total_page": page_info.get("total_page", 1),
        "reviews": formatted_reviews
    }


# ─── CREATE NEW REVIEW ──────────────────────────────────────────────
@router.post("/api/v1/reviews/create")
async def create_review(review: ReviewCreate):
    timestamp = str(int(time.time()))
    payload = {
        "product_id": str(review.product_id),
        "rating": review.rating,
        "author": review.author.strip() or "Anonymous",
        "author_country": review.author_country or "PK",
        "content": review.content.strip(),
    }
    if review.title:
        payload["title"] = review.title.strip()
    if review.author_email:
        payload["author_email"] = review.author_email.strip()

    # json.dumps must have no spaces to exactly match signature
    body_str = json.dumps(payload, separators=(",", ":"))
    
    # Generate HMAC-SHA256 signature for POST (no query params, just timestamp and body)
    sign = generate_trustoo_signature({}, body_str, timestamp)

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
        raise HTTPException(status_code=400, detail=data.get("message", "Error posting review"))
        
    return {"success": True}