import time
from typing import Literal, Optional

from .engines import BaseOcrEngine, PaddleOcrEngine, TesseractEngine
from .matcher import LabelMatcher


class OcrService:
    """Thin router that delegates OCR to pluggable engine backends."""

    ENGINES: dict[str, type[BaseOcrEngine]] = {
        "paddleocr": PaddleOcrEngine,
        "tesseract": TesseractEngine,
    }

    def __init__(
        self,
        matcher: Optional[LabelMatcher] = None,
        engines: Optional[dict[str, BaseOcrEngine]] = None,
    ):
        self.matcher = matcher or LabelMatcher()
        if engines:
            self._engines = engines
        else:
            self._engines = {
                "paddleocr": PaddleOcrEngine(),
                "tesseract": TesseractEngine(),
            }

    def get_engine(self, name: str) -> BaseOcrEngine:
        try:
            return self._engines[name]
        except KeyError:
            raise ValueError(
                f"Unknown OCR engine: {name!r}. "
                f"Available: {list(self._engines)}"
            )

    # ------------------------------------------------------------------
    # Simple OCR (single image, no evaluation)
    # ------------------------------------------------------------------

    def ocr_image(
        self,
        image_path: str,
        engine: str = "paddleocr",
        **kwargs,
    ) -> str:
        eng = self.get_engine(engine)
        img = eng.load_image(image_path)
        result = eng.ocr(img, **kwargs)
        return result["text"] if isinstance(result, dict) else result

    # ------------------------------------------------------------------
    # Attempt runner
    # ------------------------------------------------------------------

    @staticmethod
    def _run_attempt(
        eng: BaseOcrEngine,
        img,
        *,
        angle: int,
        preprocess_mode: str,
        region: str,
        psm: int = 11,
    ) -> dict:
        t0 = time.perf_counter()

        work_img = eng.select_region(img, region)
        work_img = eng.rotate_image(work_img, angle)
        result = eng.ocr(work_img, preprocess_mode=preprocess_mode, psm=psm)

        text = result["text"] if isinstance(result, dict) else result
        boxes = result.get("boxes", []) if isinstance(result, dict) else []

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "region": region,
            "angle": angle,
            "preprocess_mode": preprocess_mode,
            "psm": psm,
            "text": text,
            "boxes": boxes,
            "elapsed_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_attempts(attempts: list[dict]) -> str:
        combined = " ".join(a["text"] for a in attempts if a.get("text", "").strip())
        return " ".join(combined.split())

    @staticmethod
    def _build_ocr_by_angle(attempts: list[dict]) -> dict:
        ocr_by_angle: dict[int, str] = {}
        for a in attempts:
            if a["region"] == "full" and a["angle"] not in ocr_by_angle:
                ocr_by_angle[a["angle"]] = a["text"]
        return ocr_by_angle

    @staticmethod
    def _should_fallback(evaluation: dict) -> bool:
        fields = evaluation.get("fields", {})
        gov = evaluation.get("government_warning", {})
        critical = ["brand_name", "alcohol_content", "net_contents"]
        missing = sum(1 for f in critical if fields.get(f, {}).get("status") != "found")
        return missing > 0 or gov.get("status") == "missing"

    @staticmethod
    def _default_missing_result(ttb_id: Optional[str] = None) -> dict:
        empty_field = {
            "raw_value": None,
            "value": None,
            "normalized_value": None,
            "confidence": None,
            "status": "missing",
        }
        return {
            "id": ttb_id,
            "raw_ocr_text": "",
            "normalized_ocr_text": "",
            "fields": {
                "brand_name": dict(empty_field),
                "alcohol_content": dict(empty_field),
                "net_contents": dict(empty_field),
            },
            "government_warning": {
                "status": "missing",
                "confidence": 0.0,
                "matched_groups": {},
                "matched_anchors": [],
            },
            "warnings": ["image_not_found"],
            "ocr_by_angle": {},
            "ocr_attempts": [],
        }

    # ------------------------------------------------------------------
    # Full evaluation pipeline
    # ------------------------------------------------------------------

    def evaluate_image(
        self,
        image_path: str,
        engine: str = "paddleocr",
        ttb_id: Optional[str] = None,
        brand_name: Optional[str] = None,
        alcohol_content: Optional[str] = None,
        net_contents: Optional[str] = None,
        require_government_warning: bool = True,
        run_fallbacks: bool = True,
    ) -> dict:
        eng = self.get_engine(engine)
        img = eng.load_image(image_path)
        if img is None:
            return self._default_missing_result(ttb_id)

        attempts = []

        # First pass: full image, 0° rotation
        first = self._run_attempt(
            eng, img, angle=0, preprocess_mode="basic", region="full", psm=6,
        )
        attempts.append(first)

        combined_text = self._merge_attempts(attempts)
        evaluation = self.matcher.evaluate_text(
            raw_ocr_text=combined_text,
            ttb_id=ttb_id,
            brand_name=brand_name,
            alcohol_content=alcohol_content,
            net_contents=net_contents,
            require_government_warning=require_government_warning,
        )

        # Fallbacks — run one at a time and stop as soon as all fields are found
        if run_fallbacks and self._should_fallback(evaluation):
            for spec in eng.fallback_specs:
                attempts.append(self._run_attempt(eng, img, **spec))

                combined_text = self._merge_attempts(attempts)
                evaluation = self.matcher.evaluate_text(
                    raw_ocr_text=combined_text,
                    ttb_id=ttb_id,
                    brand_name=brand_name,
                    alcohol_content=alcohol_content,
                    net_contents=net_contents,
                    require_government_warning=require_government_warning,
                )

                if not self._should_fallback(evaluation):
                    break

        evaluation["ocr_by_angle"] = self._build_ocr_by_angle(attempts)
        evaluation["ocr_attempts"] = attempts

        # Collect bounding boxes from primary pass (full image, 0° — no coord transform needed)
        primary_boxes = []
        for a in attempts:
            if a["region"] == "full" and a["angle"] == 0:
                primary_boxes.extend(a.get("boxes", []))
        evaluation["ocr_boxes"] = primary_boxes

        return evaluation


# ------------------------------------------------------------------
# Module-level convenience functions (preserve existing public API)
# ------------------------------------------------------------------

_default_service = OcrService()


def ocr_image(
    image_path: str,
    engine: str = "paddleocr",
    **kwargs,
) -> str:
    return _default_service.ocr_image(image_path=image_path, engine=engine, **kwargs)


def evaluate_ocr_text(
    raw_ocr_text: str,
    ttb_id: Optional[str] = None,
    brand_name: Optional[str] = None,
    alcohol_content: Optional[str] = None,
    net_contents: Optional[str] = None,
    require_government_warning: bool = True,
) -> dict:
    return _default_service.matcher.evaluate_text(
        raw_ocr_text=raw_ocr_text,
        ttb_id=ttb_id,
        brand_name=brand_name,
        alcohol_content=alcohol_content,
        net_contents=net_contents,
        require_government_warning=require_government_warning,
    )


def evaluate_image(
    image_path: str,
    engine: str = "paddleocr",
    ttb_id: Optional[str] = None,
    brand_name: Optional[str] = None,
    alcohol_content: Optional[str] = None,
    net_contents: Optional[str] = None,
    require_government_warning: bool = True,
    run_fallbacks: bool = True,
) -> dict:
    return _default_service.evaluate_image(
        image_path=image_path,
        engine=engine,
        ttb_id=ttb_id,
        brand_name=brand_name,
        alcohol_content=alcohol_content,
        net_contents=net_contents,
        require_government_warning=require_government_warning,
        run_fallbacks=run_fallbacks,
    )