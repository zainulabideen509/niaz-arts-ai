"""
Product Sync Service — Pulls products from Shopify Admin API,
downloads each product image, generates CLIP + color embeddings,
and stores everything in a local JSON index file.

Run this:
  python -m scripts.sync_products

Rerun whenever you add/remove products from your Shopify store.
In production, hook this to a Shopify product webhook.
"""

import os
import io
import json
import logging
import requests
import numpy as np
from PIL import Image
from typing import Optional
from pathlib import Path

from services.clip_service import CLIPService
from services.color_service import ColorService

logger = logging.getLogger("niaz-arts-ai.sync")

# Where we store the product index
INDEX_DIR = Path("data")
INDEX_FILE = INDEX_DIR / "product_index.json"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npz"


class ProductSyncService:
    def __init__(self, clip_service: CLIPService, color_service: ColorService):
        self.clip = clip_service
        self.color = color_service

        # Shopify Admin API credentials (from .env)
        self.shop_domain = os.getenv("SHOPIFY_SHOP_DOMAIN", "niaz-arts.myshopify.com")
        self.admin_token = os.getenv("SHOPIFY_ADMIN_TOKEN", "")
        self.api_version = os.getenv("SHOPIFY_API_VERSION", "2024-10")

        if not self.admin_token:
            raise ValueError(
                "SHOPIFY_ADMIN_TOKEN not set in .env file. "
                "Go to Shopify Admin → Settings → Apps → Develop apps → "
                "your app → API credentials → Admin API access token (shpat_...)"
            )

    async def sync(self) -> int:
        """
        Pull all products from Shopify, embed them, save to disk.
        Returns the number of products synced.
        """
        logger.info("Starting product sync from Shopify...")
        products = self._fetch_all_products()
        logger.info(f"Fetched {len(products)} products from Shopify.")

        INDEX_DIR.mkdir(exist_ok=True)

        index_entries = []
        embeddings_list = []

        for i, product in enumerate(products):
            logger.info(f"Processing [{i+1}/{len(products)}]: {product['title']}")

            # Get the primary image
            image_url = None
            if product.get("images") and len(product["images"]) > 0:
                image_url = product["images"][0]["src"]
            elif product.get("image"):
                image_url = product["image"]["src"]

            if not image_url:
                logger.warning(f"  Skipping {product['title']} — no image.")
                continue

            # Download image
            try:
                img_bytes = self._download_image(image_url)
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                logger.warning(f"  Failed to download image: {e}")
                continue

            # Generate CLIP embedding
            clip_embedding = self.clip.embed_image_from_pil(pil_image)

            # Extract dominant colors
            try:
                colors = self.color.extract_colors_from_pil(pil_image, n_colors=5)
                color_data = [
                    {"hex": c.hex, "rgb": list(c.rgb), "proportion": c.proportion}
                    for c in colors
                ]
            except Exception as e:
                logger.warning(f"  Color extraction failed: {e}")
                color_data = []

            # Get price from first variant
            price = 0.0
            if product.get("variants") and len(product["variants"]) > 0:
                price = float(product["variants"][0].get("price", 0))

            # Build index entry
            entry = {
                "product_id": str(product["id"]),
                "shopify_id": f"gid://shopify/Product/{product['id']}",
                "title": product["title"],
                "description": product.get("body_html", "") or "",
                "image_url": image_url,
                "price": price,
                "vendor": product.get("vendor", ""),
                "tags": product.get("tags", "").split(", ") if isinstance(product.get("tags"), str) else product.get("tags", []),
                "colors": color_data,
                "embedding_index": len(embeddings_list),
            }

            index_entries.append(entry)
            embeddings_list.append(clip_embedding)

        # Save index as JSON
        with open(INDEX_FILE, "w") as f:
            json.dump(index_entries, f, indent=2)

        # Save embeddings as numpy compressed file
        if embeddings_list:
            embeddings_array = np.stack(embeddings_list)
            np.savez_compressed(EMBEDDINGS_FILE, embeddings=embeddings_array)

        logger.info(f"✅ Synced {len(index_entries)} products to {INDEX_FILE}")
        return len(index_entries)

    def _fetch_all_products(self) -> list:
        """Fetch all products from Shopify Admin REST API with pagination."""
        all_products = []
        url = (
            f"https://{self.shop_domain}/admin/api/{self.api_version}"
            f"/products.json?limit=250&status=active"
        )
        headers = {"X-Shopify-Access-Token": self.admin_token}

        while url:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            all_products.extend(data.get("products", []))

            # Check for pagination (Link header)
            url = None
            link_header = response.headers.get("Link", "")
            if 'rel="next"' in link_header:
                # Extract the next URL from Link header
                for part in link_header.split(","):
                    if 'rel="next"' in part:
                        url = part.split("<")[1].split(">")[0]
                        break

        return all_products

    def _download_image(self, url: str) -> bytes:
        """Download an image from URL."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
