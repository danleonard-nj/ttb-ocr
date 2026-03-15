import cv2
import pytesseract

from .base import BaseOcrEngine


class TesseractEngine(BaseOcrEngine):

    name = "tesseract"

    def __init__(self, min_confidence: int = 30):
        self._min_confidence = min_confidence

    @staticmethod
    def _preprocess(img, mode: str = "basic"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.75, fy=1.75, interpolation=cv2.INTER_CUBIC)

        if mode == "basic":
            return gray
        if mode == "median":
            return cv2.medianBlur(gray, 3)
        if mode == "bilateral":
            return cv2.bilateralFilter(gray, 5, 50, 50)
        if mode == "adaptive":
            return cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
        raise ValueError(f"Unsupported preprocess mode: {mode}")

    def ocr(self, img, *, preprocess_mode: str = "basic", psm: int = 11, **kwargs) -> dict:
        if img is None:
            return {"text": "", "boxes": []}

        gray = self._preprocess(img, mode=preprocess_mode)
        scale = 1.75
        config = f"--oem 3 --psm {psm}"

        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)

        line_words = {}
        line_bboxes = {}
        line_confs = {}
        n = len(data["text"])

        for i in range(n):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])
            if conf < self._min_confidence or not word:
                continue

            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

            if key not in line_words:
                line_words[key] = []
                line_bboxes[key] = [x, y, x + w, y + h]
                line_confs[key] = []
            else:
                b = line_bboxes[key]
                b[0] = min(b[0], x)
                b[1] = min(b[1], y)
                b[2] = max(b[2], x + w)
                b[3] = max(b[3], y + h)

            line_words[key].append(word)
            line_confs[key].append(conf)

        inv = 1.0 / scale
        all_text = " ".join(" ".join(ws) for ws in line_words.values())
        all_text = " ".join(all_text.split())

        boxes = []
        for key in line_words:
            b = line_bboxes[key]
            avg_conf = sum(line_confs[key]) / len(line_confs[key]) / 100.0
            boxes.append({
                "text": " ".join(line_words[key]),
                "bbox": [int(b[0] * inv), int(b[1] * inv), int(b[2] * inv), int(b[3] * inv)],
                "confidence": round(avg_conf, 3),
            })

        return {"text": all_text, "boxes": boxes}

    @property
    def fallback_specs(self) -> list[dict]:
        return [
            {"angle": 90, "preprocess_mode": "basic", "region": "full", "psm": 11},
            {"angle": 270, "preprocess_mode": "basic", "region": "full", "psm": 11},
            {"angle": 0, "preprocess_mode": "bilateral", "region": "full", "psm": 11},
            {"angle": 90, "preprocess_mode": "basic", "region": "right_strip", "psm": 11},
            {"angle": 270, "preprocess_mode": "basic", "region": "right_strip", "psm": 11},
            {"angle": 0, "preprocess_mode": "basic", "region": "bottom_strip", "psm": 11},
            {"angle": 0, "preprocess_mode": "adaptive", "region": "bottom_strip", "psm": 11},
        ]

    @property
    def supports_parallel(self) -> bool:
        return True
