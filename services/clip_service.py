"""
CLIP Service - Uses HuggingFace Inference API with correct endpoints.
"""

import io
import os
import logging
import requests
import numpy as np
from PIL import Image

logger = logging.getLogger("niaz-arts-ai.clip")

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_TEXT_URL = "https://api-inference.huggingface.co/models/sentence-transformers/clip-ViT-B-32"
HF_IMAGE_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"


class CLIPService:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self._local_model = None
        self._headers = {"Authorization": f"Bearer {HF_TOKEN}"}

        if not HF_TOKEN:
            logger.warning("HF_TOKEN not set! Trying local model...")
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(model_name)
                logger.info("Local CLIP model loaded.")
            except:
                logger.error("No HF_TOKEN and no local model!")
        else:
            logger.info("CLIP Service using HuggingFace API.")

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        if self._local_model:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            embedding = self._local_model.encode([image], convert_to_numpy=True)[0]
            return self._normalize(embedding)

        # Send raw image bytes to HuggingFace
        headers = {**self._headers, "Content-Type": "application/octet-stream"}
        response = requests.post(HF_IMAGE_URL, headers=headers, data=image_bytes, timeout=60)

        if response.status_code == 503:
            # Model is loading, wait and retry
            logger.info("Model loading on HuggingFace, retrying...")
            import time
            time.sleep(20)
            response = requests.post(HF_IMAGE_URL, headers=headers, data=image_bytes, timeout=60)

        response.raise_for_status()
        result = response.json()

        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                embedding = np.array(result[0], dtype=np.float32)
            else:
                embedding = np.array(result, dtype=np.float32)
        else:
            raise ValueError(f"Unexpected API response: {str(result)[:200]}")

        # CLIP image embeddings are 512-dim
        if len(embedding.shape) > 1:
            embedding = embedding.mean(axis=0)
        if embedding.shape[0] != 512:
            embedding = embedding[:512] if embedding.shape[0] > 512 else np.pad(embedding, (0, 512 - embedding.shape[0]))

        return self._normalize(embedding)

    def embed_image_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        if self._local_model:
            embedding = self._local_model.encode([pil_image], convert_to_numpy=True)[0]
            return self._normalize(embedding)

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        return self.embed_image(buf.getvalue())

    def embed_text(self, text: str) -> np.ndarray:
        if self._local_model:
            embedding = self._local_model.encode([text], convert_to_numpy=True)[0]
            return self._normalize(embedding)

        # Use sentence-transformers CLIP for text
        payload = {
            "inputs": text,
            "options": {"wait_for_model": True}
        }
        response = requests.post(HF_TEXT_URL, headers=self._headers, json=payload, timeout=60)

        if response.status_code == 503:
            logger.info("Model loading on HuggingFace, retrying...")
            import time
            time.sleep(20)
            response = requests.post(HF_TEXT_URL, headers=self._headers, json=payload, timeout=60)

        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], list):
                embedding = np.array(result[0], dtype=np.float32)
            else:
                embedding = np.array(result, dtype=np.float32)
        else:
            raise ValueError(f"Unexpected API response: {str(result)[:200]}")

        # Average token embeddings if needed
        if len(embedding.shape) > 1:
            embedding = embedding.mean(axis=0)
        if embedding.shape[0] != 512:
            embedding = embedding[:512] if embedding.shape[0] > 512 else np.pad(embedding, (0, 512 - embedding.shape[0]))

        return self._normalize(embedding)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm