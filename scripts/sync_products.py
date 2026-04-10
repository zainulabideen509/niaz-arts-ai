"""
Run this script to sync your Shopify products into the AI index.

Usage:
  python -m scripts.sync_products

This will:
1. Connect to your Shopify Admin API
2. Download all product images
3. Generate CLIP embeddings + extract dominant colors
4. Save everything to data/product_index.json and data/embeddings.npz

Rerun this whenever you add or remove products from your store.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from services.clip_service import CLIPService
from services.color_service import ColorService
from services.product_sync import ProductSyncService


async def main():
    print("=" * 60)
    print("  Niaz Arts — Product Sync")
    print("=" * 60)
    print()

    print("Step 1/3: Loading CLIP model...")
    clip = CLIPService()
    print("  ✅ CLIP model loaded.")

    print("\nStep 2/3: Initializing color service...")
    color = ColorService()
    print("  ✅ Color service ready.")

    print("\nStep 3/3: Syncing products from Shopify...")
    sync = ProductSyncService(clip, color)
    count = await sync.sync()

    print()
    print("=" * 60)
    print(f"  ✅ Done! {count} products synced and indexed.")
    print("  You can now start the API server: python main.py")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
