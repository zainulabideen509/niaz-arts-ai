"""
Niaz Arts - Product Sync from CSV Export
No API token needed!
"""

import csv
import io
import sys
import os
import json
import logging
import requests
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.clip_service import CLIPService
from services.color_service import ColorService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sync")

INDEX_DIR = Path("data")
INDEX_FILE = INDEX_DIR / "product_index.json"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npz"


def safe(val):
    if val is None:
        return ""
    return str(val).strip()


def download_image(url, timeout=30):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.warning(f"  Failed to download: {e}")
        return None


def main():
    csv_path = "products.csv"

    if not os.path.exists(csv_path):
        print()
        print("ERROR: products.csv not found!")
        print("Export from Shopify Admin -> Products -> Export")
        return

    print("=" * 60)
    print("  Niaz Arts - Product Sync from CSV")
    print("=" * 60)

    print("\nStep 1/3: Loading CLIP model...")
    clip = CLIPService()
    print("  CLIP model loaded.")

    print("\nStep 2/3: Initializing color service...")
    color = ColorService()
    print("  Color service ready.")

    print("\nStep 3/3: Processing products from CSV...")

    products = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            handle = safe(row.get("Handle"))
            if not handle:
                continue

            if handle not in products:
                title = safe(row.get("Title"))
                if not title:
                    continue

                image_url = safe(row.get("Image Src"))
                price = safe(row.get("Variant Price"))
                vendor = safe(row.get("Vendor"))
                tags = safe(row.get("Tags"))
                description = safe(row.get("Body (HTML)"))
                product_type = safe(row.get("Type"))
                status = safe(row.get("Status")).lower()

                if status and status != "active":
                    continue

                try:
                    price_float = float(price) if price else 0.0
                except ValueError:
                    price_float = 0.0

                tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

                products[handle] = {
                    "handle": handle,
                    "title": title,
                    "description": description,
                    "image_url": image_url,
                    "price": price_float,
                    "vendor": vendor,
                    "tags": tag_list,
                    "type": product_type,
                }

    print(f"  Found {len(products)} unique products in CSV.")

    if not products:
        print("\n  No products found! Check your CSV file.")
        return

    INDEX_DIR.mkdir(exist_ok=True)
    index_entries = []
    embeddings_list = []

    product_list = list(products.values())
    for i, product in enumerate(product_list):
        title = product["title"]
        image_url = product["image_url"]

        print(f"  Processing [{i+1}/{len(product_list)}]: {title}")

        if not image_url:
            print(f"    Skipping - no image.")
            continue

        img_bytes = download_image(image_url)
        if not img_bytes:
            continue

        try:
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            print(f"    Could not open image: {e}")
            continue

        clip_embedding = clip.embed_image_from_pil(pil_image)

        try:
            colors = color.extract_colors_from_pil(pil_image, n_colors=5)
            color_data = [
                {"hex": c.hex, "rgb": list(c.rgb), "proportion": c.proportion}
                for c in colors
            ]
        except Exception as e:
            print(f"    Color extraction failed: {e}")
            color_data = []

        entry = {
            "product_id": product["handle"],
            "shopify_id": product["handle"],
            "title": title,
            "description": product["description"],
            "image_url": image_url,
            "price": product["price"],
            "vendor": product["vendor"],
            "tags": product["tags"],
            "colors": color_data,
            "embedding_index": len(embeddings_list),
        }

        index_entries.append(entry)
        embeddings_list.append(clip_embedding)
        print(f"    Done - {len(color_data)} colors extracted.")

    with open(INDEX_FILE, "w") as f:
        json.dump(index_entries, f, indent=2)

    if embeddings_list:
        embeddings_array = np.stack(embeddings_list)
        np.savez_compressed(EMBEDDINGS_FILE, embeddings=embeddings_array)

    print()
    print("=" * 60)
    print(f"  Done! {len(index_entries)} products synced and indexed.")
    print(f"  You can now start the AI server: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()