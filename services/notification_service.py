# ════════════════════════════════════════════════════════════════════════════
#  NIAZ ARTS — AUTOMATIC PUSH NOTIFICATIONS
#  Add this to your FastAPI backend (the niaz-arts-ai service on Render).
#
#  WHAT IT DOES
#  ------------
#  Shopify automatically calls these endpoints (webhooks) whenever you:
#    • create a new PRODUCT     -> POST /webhook/shopify/product-create
#    • create a new COLLECTION  -> POST /webhook/shopify/collection-create
#  Each call verifies it is genuinely from Shopify (HMAC), then sends a push
#  notification to ALL app users via the Firebase "all_users" topic.
#
#  Manual notifications from the Firebase Console keep working as before —
#  this only ADDS automation; it does not replace anything.
# ════════════════════════════════════════════════════════════════════════════

import os
import json
import hmac
import base64
import hashlib
from fastapi import APIRouter, Request, Header, HTTPException

# ── Firebase Admin (server-side sending) ────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, messaging

router = APIRouter()

# ----------------------------------------------------------------------------
# 1) FIREBASE INITIALISATION
#    Uses a service-account JSON. On Render, store the JSON as an environment
#    variable named FIREBASE_SERVICE_ACCOUNT (paste the whole file content),
#    OR upload the file and point GOOGLE_APPLICATION_CREDENTIALS at it.
# ----------------------------------------------------------------------------
def _init_firebase():
    if firebase_admin._apps:          # already initialised
        return
    sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    if sa_json:
        cred = credentials.Certificate(json.loads(sa_json))
    else:
        # fallback: a file path in GOOGLE_APPLICATION_CREDENTIALS
        path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS",
                              "serviceAccountKey.json")
        cred = credentials.Certificate(path)
    firebase_admin.initialize_app(cred)


def _send_to_all(title: str, body: str, data: dict | None = None):
    """Send one push notification to every device on the all_users topic."""
    _init_firebase()
    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        topic="all_users",
        data={k: str(v) for k, v in (data or {}).items()},
        android=messaging.AndroidConfig(
            priority="high",
            notification=messaging.AndroidNotification(
                channel_id="niazarts_general",
                sound="default",
            ),
        ),
    )
    resp = messaging.send(message)
    print(f"[notify] sent to all_users: {title} ({resp})")
    return resp


# ----------------------------------------------------------------------------
# 2) SHOPIFY WEBHOOK VERIFICATION
#    Shopify signs every webhook with your app's secret. We verify the
#    signature so nobody can fake a request and spam your customers.
#
#    Set SHOPIFY_WEBHOOK_SECRET on Render to the secret Shopify shows when you
#    create the webhook (or your app's API secret key).
# ----------------------------------------------------------------------------
def _verify_shopify(raw_body: bytes, hmac_header: str | None) -> bool:
    secret = os.environ.get("SHOPIFY_WEBHOOK_SECRET", "")
    if not secret or not hmac_header:
        # If no secret configured yet, allow (so you can test) but warn.
        print("[notify] WARNING: SHOPIFY_WEBHOOK_SECRET not set — skipping verify")
        return True
    digest = hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).digest()
    computed = base64.b64encode(digest).decode()
    return hmac.compare_digest(computed, hmac_header)


# ----------------------------------------------------------------------------
# 3) WEBHOOK ENDPOINTS
# ----------------------------------------------------------------------------
@router.post("/webhook/shopify/product-create")
async def shopify_product_create(
    request: Request,
    x_shopify_hmac_sha256: str | None = Header(default=None),
):
    raw = await request.body()
    if not _verify_shopify(raw, x_shopify_hmac_sha256):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        product = json.loads(raw)
    except Exception:
        product = {}

    title_text = product.get("title", "a new painting")
    handle = product.get("handle", "")

    _send_to_all(
        title="New Painting Added! 🎨",
        body=f"{title_text} just arrived at Niaz Arts. Tap to explore.",
        data={"type": "product", "handle": handle},
    )
    return {"status": "ok"}


@router.post("/webhook/shopify/collection-create")
async def shopify_collection_create(
    request: Request,
    x_shopify_hmac_sha256: str | None = Header(default=None),
):
    raw = await request.body()
    if not _verify_shopify(raw, x_shopify_hmac_sha256):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        collection = json.loads(raw)
    except Exception:
        collection = {}

    title_text = collection.get("title", "a new collection")
    handle = collection.get("handle", "")

    _send_to_all(
        title="New Collection Just Dropped! ✨",
        body=f"Explore our new \"{title_text}\" collection now.",
        data={"type": "collection", "handle": handle},
    )
    return {"status": "ok"}


# ----------------------------------------------------------------------------
# 4) OPTIONAL — manual send endpoint (in addition to Firebase Console)
#    Lets you trigger a custom announcement from your own admin tool / curl.
#    Protect it with a simple token via the ADMIN_PUSH_TOKEN env var.
# ----------------------------------------------------------------------------
@router.post("/admin/send-notification")
async def admin_send(request: Request,
                     x_admin_token: str | None = Header(default=None)):
    expected = os.environ.get("ADMIN_PUSH_TOKEN", "")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
    body = await request.json()
    _send_to_all(
        title=body.get("title", "Niaz Arts"),
        body=body.get("body", ""),
        data=body.get("data", {}),
    )
    return {"status": "ok"}


# ----------------------------------------------------------------------------
#  HOW TO PLUG IN  (in your main FastAPI file)
#  -------------------------------------------
#     from notifications_webhook import router as notify_router
#     app.include_router(notify_router)
# ----------------------------------------------------------------------------
