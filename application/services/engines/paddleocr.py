import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import threading

import cv2

from .base import BaseOcrEngine


class PaddleOcrEngine(BaseOcrEngine):

    name = "paddleocr"

    def __init__(self, lang: str = "en"):
        self._lang = lang
        self._instance = None
        self._lock = threading.Lock()

    def _get_instance(self):
        if self._instance is None:
            from paddleocr import PaddleOCR
            self._instance = PaddleOCR(
                ocr_version="PP-OCRv4",
                lang=self._lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_hpi=False,
            )
        return self._instance

    def ocr(self, img, **kwargs) -> dict:
        if img is None:
            return {"text": "", "boxes": []}

        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 1600:
            scale = 1600 / max_dim
            work = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            work = img

        with self._lock:
            ocr = self._get_instance()
            results = ocr.predict(work)

        lines = []
        boxes = []
        for res in results:
            if isinstance(res, dict):
                rec_texts = res.get("rec_texts", [])
                dt_polys = res.get("dt_polys", [])
                rec_scores = res.get("rec_scores", [])
            elif hasattr(res, "res") and isinstance(res.res, dict):
                rec_texts = res.res.get("rec_texts", [])
                dt_polys = res.res.get("dt_polys", [])
                rec_scores = res.res.get("rec_scores", [])
            else:
                rec_texts = getattr(res, "rec_texts", [])
                dt_polys = getattr(res, "dt_polys", [])
                rec_scores = getattr(res, "rec_scores", [])

            for i, text in enumerate(rec_texts or []):
                if not text.strip():
                    continue
                lines.append(text.strip())
                if dt_polys is not None and i < len(dt_polys):
                    try:
                        poly = dt_polys[i]
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        inv = 1.0 / scale
                        boxes.append({
                            "text": text.strip(),
                            "bbox": [
                                int(min(xs) * inv),
                                int(min(ys) * inv),
                                int(max(xs) * inv),
                                int(max(ys) * inv),
                            ],
                            "confidence": round(float(rec_scores[i]), 3) if rec_scores and i < len(rec_scores) else 0.0,
                        })
                    except (TypeError, IndexError):
                        pass

        return {"text": " ".join(lines), "boxes": boxes}

    @property
    def fallback_specs(self) -> list[dict]:
        return [
            {"angle": 90, "preprocess_mode": "basic", "region": "full", "psm": 11},
            {"angle": 270, "preprocess_mode": "basic", "region": "full", "psm": 11},
            {"angle": 0, "preprocess_mode": "basic", "region": "bottom_strip", "psm": 11},
        ]

    @property
    def supports_parallel(self) -> bool:
        return False
