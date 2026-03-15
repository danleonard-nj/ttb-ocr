from abc import ABC, abstractmethod

import cv2


class BaseOcrEngine(ABC):
    """Abstract base for OCR engines."""

    name: str = "base"

    @abstractmethod
    def ocr(self, img, **kwargs) -> str:
        """Run OCR on a BGR numpy image and return extracted text."""

    @property
    def fallback_specs(self) -> list[dict]:
        """Return the list of fallback attempt specs for this engine."""
        return []

    @property
    def supports_parallel(self) -> bool:
        """Whether fallback attempts can be run in parallel."""
        return False

    # ------------------------------------------------------------------
    # Shared image utilities
    # ------------------------------------------------------------------

    @staticmethod
    def load_image(image_path: str):
        return cv2.imread(image_path)

    @staticmethod
    def rotate_image(img, angle: int):
        if angle == 0:
            return img
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError(f"Unsupported angle: {angle}")

    @staticmethod
    def crop_right_strip(img, fraction: float = 0.28):
        if img is None:
            return None
        h, w = img.shape[:2]
        x0 = max(0, int(w * (1.0 - fraction)))
        return img[:, x0:w]

    @staticmethod
    def crop_bottom_strip(img, fraction: float = 0.28):
        if img is None:
            return None
        h, w = img.shape[:2]
        y0 = max(0, int(h * (1.0 - fraction)))
        return img[y0:h, :]

    def select_region(self, img, region: str):
        if region == "full":
            return img
        if region == "right_strip":
            return self.crop_right_strip(img)
        if region == "bottom_strip":
            return self.crop_bottom_strip(img)
        raise ValueError(f"Unsupported region: {region}")
