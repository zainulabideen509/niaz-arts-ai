"""
Recommendation Service — The brain that combines signals.

For IMAGE recommendations (wall photo upload):
  Score = 0.55 × CLIP_similarity + 0.45 × Color_similarity
  
  - CLIP captures: "this wall has a modern minimalist vibe" → match abstract art
  - Color captures: "this wall is warm beige" → match paintings with warm tones
  
For TEXT recommendations ("calm blue painting for bedroom"):
  Score = CLIP_text_similarity (pure semantic match)
  
  CLIP already understands colors, moods, and styles from text,
  so color extraction isn't needed for text queries.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from services.clip_service import CLIPService
from services.color_service import ColorService, DominantColor

logger = logging.getLogger("niaz-arts-ai.recommend")

INDEX_FILE = Path("data/product_index.json")
EMBEDDINGS_FILE = Path("data/embeddings.npz")

# Weights for combining signals (image recommendation)
CLIP_WEIGHT = 0.55
COLOR_WEIGHT = 0.45


class RecommendationService:
    def __init__(self, clip_service: CLIPService, color_service: ColorService):
        self.clip = clip_service
        self.color = color_service
        self.products: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None

        self.reload_index()

    def reload_index(self):
        """Load product index and embeddings from disk."""
        if INDEX_FILE.exists() and EMBEDDINGS_FILE.exists():
            with open(INDEX_FILE) as f:
                self.products = json.load(f)
            data = np.load(EMBEDDINGS_FILE)
            self.embeddings = data["embeddings"]
            logger.info(f"Loaded {len(self.products)} products, embeddings shape: {self.embeddings.shape}")
        else:
            self.products = []
            self.embeddings = None
            logger.warning("No product index found.")

    def has_products(self) -> bool:
        return len(self.products) > 0 and self.embeddings is not None

    def product_count(self) -> int:
        return len(self.products)

    def recommend_by_image(
        self, image_bytes: bytes, top_k: int = 6
    ) -> Tuple[List[dict], List[str]]:
        """
        Given a wall/room photo, return top-K matching paintings.
        
        Returns: (list of recommendation dicts, list of wall color hex strings)
        """
        # 1. Embed the wall image with CLIP
        wall_embedding = self.clip.embed_image(image_bytes)

        # 2. Extract wall colors
        wall_colors = self.color.extract_colors(image_bytes, n_colors=5)
        wall_color_hexes = [c.hex for c in wall_colors]

        # 3. Calculate CLIP similarity for all products
        clip_similarities = np.dot(self.embeddings, wall_embedding)

        # 4. Calculate color similarity for all products
        color_similarities = np.zeros(len(self.products))
        for i, product in enumerate(self.products):
            product_colors = [
                DominantColor(
                    hex=c["hex"],
                    rgb=tuple(c["rgb"]),
                    proportion=c["proportion"],
                )
                for c in product.get("colors", [])
            ]
            if product_colors:
                color_similarities[i] = self.color.color_similarity(
                    wall_colors, product_colors
                )

        # 5. Combine scores
        combined_scores = (
            CLIP_WEIGHT * clip_similarities + COLOR_WEIGHT * color_similarities
        )

        # 6. Get top-K indices
        top_indices = np.argsort(-combined_scores)[:top_k]

        # 7. Build results
        results = []
        for idx in top_indices:
            product = self.products[idx]
            clip_score = float(clip_similarities[idx])
            color_score = float(color_similarities[idx])
            combined = float(combined_scores[idx])

            # Generate a human-readable match reason
            reason = self._generate_match_reason(clip_score, color_score, product)

            results.append({
                "product_id": product["product_id"],
                "shopify_id": product["shopify_id"],
                "title": product["title"],
                "image_url": product["image_url"],
                "price": product["price"],
                "score": round(combined, 4),
                "match_reason": reason,
            })

        return results, wall_color_hexes

    def recommend_by_text(self, query: str, top_k: int = 6) -> List[dict]:
        """
        Given a text description, return top-K matching paintings.
        Pure CLIP semantic similarity — no color extraction needed.
        """
        # 1. Embed the text query with CLIP
        text_embedding = self.clip.embed_text(query)

        # 2. Calculate similarity against all product embeddings
        similarities = np.dot(self.embeddings, text_embedding)

        # 3. Get top-K
        top_indices = np.argsort(-similarities)[:top_k]

        # 4. Build results
        results = []
        for idx in top_indices:
            product = self.products[idx]
            score = float(similarities[idx])

            results.append({
                "product_id": product["product_id"],
                "shopify_id": product["shopify_id"],
                "title": product["title"],
                "image_url": product["image_url"],
                "price": product["price"],
                "score": round(score, 4),
                "match_reason": f"Matches your description: '{query}'",
            })

        return results

    def _generate_match_reason(
        self, clip_score: float, color_score: float, product: dict
    ) -> str:
        """Generate a simple explanation of why this painting was recommended."""
        reasons = []

        if clip_score > 0.25:
            reasons.append("style matches your space")
        if color_score > 0.6:
            reasons.append("color palette complements your wall")
        elif color_score > 0.4:
            reasons.append("colors work well with your room")

        if not reasons:
            reasons.append("good overall match for your space")

        return " & ".join(reasons).capitalize()
