"""
Color Service — Extracts dominant colors from images.

Used for two things:
1. Analyze the customer's wall photo → extract wall colors
2. During product sync → extract dominant colors from each painting

Then we match wall colors against painting colors as one signal
(combined with CLIP semantic similarity for the final ranking).

Uses OpenCV for image processing and scikit-learn KMeans for color clustering.
This satisfies your SRS requirement FR-2.2 (extract top 3-5 dominant colors).
"""

import io
import logging
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger("niaz-arts-ai.color")


@dataclass
class DominantColor:
    """A dominant color with its hex value and proportion in the image."""
    hex: str
    rgb: Tuple[int, int, int]
    proportion: float  # 0.0 to 1.0


class ColorService:
    def __init__(self, n_colors: int = 5):
        self.n_colors = n_colors

    def extract_colors(self, image_bytes: bytes, n_colors: int = None) -> List[DominantColor]:
        """
        Extract dominant colors from an image using KMeans clustering.
        Returns colors sorted by proportion (most dominant first).
        """
        n = n_colors or self.n_colors

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        # Convert BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize for speed (KMeans on a 4000x3000 image is slow)
        img_small = cv2.resize(img_rgb, (150, 150))

        # Reshape to a list of pixels: (22500, 3)
        pixels = img_small.reshape(-1, 3).astype(np.float32)

        # Run KMeans clustering
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Get cluster centers (the dominant colors) and their proportions
        centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        proportions = counts / len(labels)

        # Sort by proportion (most dominant first)
        sorted_indices = np.argsort(-proportions)

        colors = []
        for idx in sorted_indices:
            r, g, b = int(centers[idx][0]), int(centers[idx][1]), int(centers[idx][2])
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            colors.append(DominantColor(
                hex=hex_color,
                rgb=(r, g, b),
                proportion=float(proportions[idx]),
            ))

        return colors

    def extract_colors_from_pil(self, pil_image: Image.Image, n_colors: int = None) -> List[DominantColor]:
        """Extract colors from a PIL Image (used during product sync)."""
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG")
        return self.extract_colors(buf.getvalue(), n_colors)

    def color_similarity(
        self,
        colors_a: List[DominantColor],
        colors_b: List[DominantColor],
    ) -> float:
        """
        Calculate similarity between two color palettes.

        Uses a weighted minimum-distance approach in LAB color space
        (which is perceptually uniform — unlike RGB, equal numeric distance
        = equal perceived difference).

        Returns a score from 0.0 (totally different) to 1.0 (identical palettes).
        """
        if not colors_a or not colors_b:
            return 0.0

        # Convert to LAB for perceptually meaningful distance
        lab_a = [self._rgb_to_lab(c.rgb) for c in colors_a]
        lab_b = [self._rgb_to_lab(c.rgb) for c in colors_b]
        weights_a = [c.proportion for c in colors_a]

        total_distance = 0.0
        total_weight = 0.0

        for i, (la, wa) in enumerate(zip(lab_a, weights_a)):
            # Find minimum distance to any color in palette B
            min_dist = min(self._lab_distance(la, lb) for lb in lab_b)
            total_distance += wa * min_dist
            total_weight += wa

        if total_weight == 0:
            return 0.0

        avg_distance = total_distance / total_weight

        # Normalize: max meaningful distance in LAB is ~100-150
        # Convert to 0-1 similarity score
        max_distance = 120.0
        similarity = max(0.0, 1.0 - (avg_distance / max_distance))
        return similarity

    @staticmethod
    def _rgb_to_lab(rgb: Tuple[int, int, int]) -> np.ndarray:
        """Convert RGB to CIELAB color space using OpenCV."""
        pixel = np.uint8([[list(rgb)]])
        lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)
        return lab[0][0].astype(np.float32)

    @staticmethod
    def _lab_distance(lab1: np.ndarray, lab2: np.ndarray) -> float:
        """Euclidean distance in LAB space (≈ Delta E)."""
        return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))
