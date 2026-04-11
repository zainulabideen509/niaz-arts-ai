"""
CLIP Service - Lightweight version for free hosting.
Uses pre-computed embeddings for products.
For user queries: color matching (images) + keyword matching (text).
No heavy model or external API needed on the server.
"""

import io
import os
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger("niaz-arts-ai.clip")


class CLIPService:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        logger.info("CLIP Service initialized (lightweight mode).")

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        """Return a zero vector - image matching uses color service instead."""
        return np.zeros(512, dtype=np.float32)

    def embed_image_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        return np.zeros(512, dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Return a zero vector - text matching uses keyword service instead."""
        return np.zeros(512, dtype=np.float32)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm