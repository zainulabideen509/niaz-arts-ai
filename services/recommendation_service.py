"""
Recommendation Service - Works without CLIP model on server.
Image recommendations: Pure color matching (very effective for wall-to-painting)
Text recommendations: Keyword matching against product titles, tags, descriptions
"""

import json
import logging
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from services.clip_service import CLIPService
from services.color_service import ColorService, DominantColor

logger = logging.getLogger("niaz-arts-ai.recommend")

INDEX_FILE = Path("data/product_index.json")
EMBEDDINGS_FILE = Path("data/embeddings.npz")


class RecommendationService:
    def __init__(self, clip_service: CLIPService, color_service: ColorService):
        self.clip = clip_service
        self.color = color_service
        self.products: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None

        self.reload_index()

    def reload_index(self):
        if INDEX_FILE.exists():
            with open(INDEX_FILE) as f:
                self.products = json.load(f)
            if EMBEDDINGS_FILE.exists():
                data = np.load(EMBEDDINGS_FILE)
                self.embeddings = data["embeddings"]
            logger.info(f"Loaded {len(self.products)} products.")
        else:
            self.products = []
            self.embeddings = None
            logger.warning("No product index found.")

    def has_products(self) -> bool:
        return len(self.products) > 0

    def product_count(self) -> int:
        return len(self.products)

    def recommend_by_image(
        self, image_bytes: bytes, top_k: int = 6
    ) -> Tuple[List[dict], List[str]]:
        """
        Image recommendation using COLOR MATCHING.
        Extracts colors from the wall photo and finds paintings
        with complementary/matching color palettes.
        """
        # Extract wall colors
        wall_colors = self.color.extract_colors(image_bytes, n_colors=5)
        wall_color_hexes = [c.hex for c in wall_colors]

        # Calculate color similarity for all products
        scores = []
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
                color_score = self.color.color_similarity(wall_colors, product_colors)
            else:
                color_score = 0.0
            scores.append(color_score)

        scores = np.array(scores)

        # Get top-K indices
        top_indices = np.argsort(-scores)[:top_k]

        # Build results
        results = []
        for idx in top_indices:
            product = self.products[idx]
            score = float(scores[idx])

            if score < 0.01:
                continue

            reason = self._color_match_reason(score)

            results.append({
                "product_id": product["product_id"],
                "shopify_id": product["shopify_id"],
                "title": product["title"],
                "image_url": product["image_url"],
                "price": product["price"],
                "score": round(score, 4),
                "match_reason": reason,
            })

        return results, wall_color_hexes

    def recommend_by_text(self, query: str, top_k: int = 6) -> List[dict]:
        """
        Text recommendation using KEYWORD MATCHING.
        Matches user query against product titles, tags, and descriptions.
        """
        query_lower = query.lower().strip()
        query_words = set(re.findall(r'\w+', query_lower))

        # Remove common stop words
        stop_words = {'a', 'an', 'the', 'for', 'my', 'i', 'want', 'need',
                      'looking', 'find', 'me', 'show', 'get', 'would', 'like',
                      'something', 'painting', 'paintings', 'art', 'artwork',
                      'wall', 'room', 'with', 'in', 'on', 'to', 'of', 'and',
                      'that', 'is', 'are', 'it', 'be', 'can', 'please'}
        query_words = query_words - stop_words

        if not query_words:
            query_words = set(re.findall(r'\w+', query_lower))

        scores = []
        for product in self.products:
            score = self._text_match_score(query_words, query_lower, product)
            scores.append(score)

        scores = np.array(scores)

        # If no keyword matches found, try fuzzy matching on all products
        if scores.max() == 0:
            for i, product in enumerate(self.products):
                title_lower = product.get("title", "").lower()
                if any(w in title_lower for w in query_words):
                    scores[i] = 0.3

        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_indices:
            product = self.products[idx]
            score = float(scores[idx])

            if score < 0.01:
                continue

            results.append({
                "product_id": product["product_id"],
                "shopify_id": product["shopify_id"],
                "title": product["title"],
                "image_url": product["image_url"],
                "price": product["price"],
                "score": round(min(score, 1.0), 4),
                "match_reason": f"Matches your description: '{query}'",
            })

        return results

    def _text_match_score(self, query_words: set, query_lower: str, product: dict) -> float:
        """Calculate how well a product matches the text query."""
        score = 0.0

        title = product.get("title", "").lower()
        tags = [t.lower() for t in product.get("tags", [])]
        description = product.get("description", "").lower()
        vendor = product.get("vendor", "").lower()

        title_words = set(re.findall(r'\w+', title))
        tag_words = set()
        for tag in tags:
            tag_words.update(re.findall(r'\w+', tag))
        desc_words = set(re.findall(r'\w+', description))

        # Title matches (highest weight)
        title_matches = query_words & title_words
        score += len(title_matches) * 0.4

        # Tag matches (high weight)
        tag_matches = query_words & tag_words
        score += len(tag_matches) * 0.35

        # Description matches (medium weight)
        desc_matches = query_words & desc_words
        score += len(desc_matches) * 0.1

        # Vendor match
        if any(w in vendor for w in query_words):
            score += 0.15

        # Bonus for exact phrase match in title
        if query_lower in title:
            score += 0.5

        # Bonus for partial matches in tags
        for tag in tags:
            if query_lower in tag or tag in query_lower:
                score += 0.3

        # Color-related keywords
        color_keywords = {
            'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
            'black', 'white', 'grey', 'gray', 'brown', 'gold', 'golden',
            'silver', 'beige', 'navy', 'teal', 'maroon', 'cream', 'ivory'
        }
        color_matches = query_words & color_keywords
        for color in color_matches:
            if color in title or any(color in t for t in tags):
                score += 0.3

        # Style-related keywords
        style_map = {
            'abstract': ['abstract', 'modern', 'contemporary'],
            'calligraphy': ['calligraphy', 'islamic', 'arabic', 'quran', 'ayat', 'bismillah'],
            'landscape': ['landscape', 'nature', 'mountain', 'sea', 'ocean', 'forest', 'tree'],
            'floral': ['floral', 'flower', 'flowers', 'rose', 'botanical'],
            'modern': ['modern', 'contemporary', 'minimalist', 'minimal'],
            'traditional': ['traditional', 'classic', 'classical', 'vintage'],
            'portrait': ['portrait', 'face', 'figure', 'people'],
        }

        for style, keywords in style_map.items():
            if any(k in query_lower for k in keywords):
                if any(k in title for k in keywords) or any(any(k in t for k in keywords) for t in tags):
                    score += 0.4

        # Mood-related keywords
        mood_map = {
            'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'relaxing', 'soothing'],
            'bold': ['bold', 'vibrant', 'bright', 'colorful', 'vivid', 'energetic'],
            'warm': ['warm', 'cozy', 'earthy', 'autumn', 'sunset'],
            'cool': ['cool', 'cold', 'winter', 'ice', 'frost'],
            'elegant': ['elegant', 'luxury', 'premium', 'masterpiece', 'exclusive'],
        }

        for mood, keywords in mood_map.items():
            if any(k in query_lower for k in keywords):
                if any(k in title for k in keywords) or any(any(k in t for k in keywords) for t in tags):
                    score += 0.25

        return score

    def _color_match_reason(self, score: float) -> str:
        if score > 0.7:
            return "Color palette perfectly matches your space"
        elif score > 0.5:
            return "Colors complement your wall beautifully"
        elif score > 0.3:
            return "Color tones work well with your room"
        else:
            return "Good color match for your space"