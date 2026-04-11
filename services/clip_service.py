"""
CLIP Service - Uses HuggingFace Inference API.
Keeps memory under 512MB for free hosting.
"""

import io
import os
import base64
import logging
import requests
import numpy as np
from PIL import Image

logger = logging.getLogger("niaz-arts-ai.clip")

HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/openai/clip-vit-base-patch32"
HF_TOKEN = os.getenv("HF_TOKEN", "")


class CLIPService:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        if not HF_TOKEN:
            logger.warning("HF_TOKEN not set! Trying local model...")
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(model_name)
                logger.info("Local CLIP model loaded.")
            except:
                logger.error("No HF_TOKEN and no local model available!")
                self._local_model = None
        else:
            logger.info("CLIP Service using HuggingFace API.")
            self._local_model = None

        self._headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    def embed_image(self, image_bytes: bytes) -> np.ndarray:
        if self._local_model:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            embedding = self._local_model.encode([image], convert_to_numpy=True)[0]
            return self._normalize(embedding)

        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload = {"inputs": {"image": img_b64}}
        response = requests.post(HF_API_URL, headers=self._headers, json=payload, timeout=30)
        response.raise_for_status()
        embedding = np.array(response.json()[0], dtype=np.float32)
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

        payload = {"inputs": text}
        response = requests.post(HF_API_URL, headers=self._headers, json=payload, timeout=30)
        response.raise_for_status()
        embedding = np.array(response.json()[0], dtype=np.float32)
        return self._normalize(embedding)

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        return float(np.dot(vec_a, vec_b))

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm