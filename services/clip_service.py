"""
CLIP Service — Embeds images and text into the same vector space.

Uses OpenAI's CLIP model via sentence-transformers.
This single model powers BOTH image and text recommendations:
- Upload a wall photo → CLIP embeds it → find similar product embeddings
- Type "calm blue abstract" → CLIP embeds text → find similar product embeddings

Why CLIP and not a custom model?
- Zero training needed (pretrained on 400M image-text pairs)
- Understands artistic concepts, colors, moods, styles out of the box
- Embeds images AND text into the SAME space — one model, two features
- This is your answer at the viva when they ask "how does it work?"
"""

import io
import logging
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("niaz-arts-ai.clip")


class CLIPService:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Load the CLIP model. First run downloads ~350MB, then it's cached.
        """
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = SentenceTransformer("clip-ViT-B-32")
        logger.info("CLIP model loaded successfully.")

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Convert raw image bytes into a 512-dimensional CLIP embedding vector.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        embedding = self.model.encode([image], convert_to_numpy=True)[0]
        return self._normalize(embedding)

    def embed_image_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """
        Embed a PIL Image directly (used during product sync).
        """
        embedding = self.model.encode([pil_image], convert_to_numpy=True)[0]
        return self._normalize(embedding)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a text query into a 512-dimensional CLIP embedding vector.
        CLIP was trained on image-text pairs, so this embedding lives in the
        SAME space as image embeddings — that's the magic.
        """
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        return self._normalize(embedding)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity between two normalized vectors."""
        return float(np.dot(vec_a, vec_b))

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm
