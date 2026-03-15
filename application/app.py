import os
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import asyncio
import logging
import tempfile
import time

import cv2
import numpy as np
import pytesseract
from quart import Quart, render_template, request, jsonify

from services.ocr import OcrService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

app = Quart(__name__)
ocr_service = OcrService()

STATUS_MAP = {
    "found": "match",
    "partial": "partial_match",
    "missing": "not_found",
}


def _warm_tesseract() -> None:
    t_start = time.perf_counter()
    dummy = np.zeros((100, 100), dtype=np.uint8)
    _ = cv2.GaussianBlur(dummy, (3, 3), 0)
    pytesseract.image_to_string(dummy)
    elapsed_ms = round((time.perf_counter() - t_start) * 1000)
    log.info("Tesseract warmup complete in %dms", elapsed_ms)


@app.before_serving
async def warmup_ocr_engines() -> None:
    """Kick off warmup in background so the server is immediately live."""
    asyncio.get_event_loop().run_in_executor(None, _warm_tesseract)


def _translate_field(field: dict) -> dict:
    """Translate OCR service field shape to the frontend API contract."""
    return {
        "expected": field.get("value"),
        "extracted": field.get("raw_value"),
        "confidence": field.get("confidence") or 0.0,
        "match_status": STATUS_MAP.get(field.get("status"), "not_found"),
    }


def _translate_gov_warning(gov: dict, expected: bool) -> dict:
    return {
        "expected": expected,
        "extracted": gov.get("status") == "found",
        "confidence": gov.get("confidence", 0.0),
        "match_status": STATUS_MAP.get(gov.get("status"), "not_found"),
    }


def _box_match_score(raw_value: str, box_text: str, field_name: str) -> float:
    """Score how well a single OCR box matches a field's raw value (0.0–1.0)."""
    if not raw_value or not box_text:
        return 0.0

    # Exact containment: raw value fully inside box text
    if raw_value in box_text:
        return 1.0

    # For numeric fields, collapse whitespace before comparing so
    # "750ml" matches a box reading "750 ML"
    if field_name in ("alcohol_content", "net_contents"):
        rv_compact = raw_value.replace(" ", "")
        bt_compact = box_text.replace(" ", "")
        if rv_compact in bt_compact or bt_compact in rv_compact:
            return 0.95

    # Reverse containment: box text inside raw value (e.g. box="cascade winery"
    # inside raw="cascade winery").  Require significant coverage so a box with
    # just one short word (e.g. "red") doesn't match.
    if len(box_text) > 2 and box_text in raw_value:
        coverage = len(box_text) / len(raw_value)
        if coverage >= 0.4:
            return coverage * 0.9

    # Token overlap — require at least half the expected tokens to be present
    raw_tokens = set(raw_value.split())
    bt_tokens = set(box_text.split())
    if raw_tokens and bt_tokens:
        recall = len(raw_tokens & bt_tokens) / len(raw_tokens)
        if recall >= 0.5:
            return recall * 0.7

    return 0.0


def _associate_field_boxes(ocr_result: dict) -> dict:
    """Find bounding boxes from the primary OCR pass that correspond to each field."""
    all_boxes = ocr_result.get("ocr_boxes", [])
    field_boxes = {}

    for field_name, field in ocr_result.get("fields", {}).items():
        if field.get("status") == "missing" or not field.get("raw_value"):
            field_boxes[field_name] = []
            continue

        raw_value = str(field["raw_value"]).lower().strip()

        scored = []
        for box in all_boxes:
            bt = box["text"].lower().strip()
            score = _box_match_score(raw_value, bt, field_name)
            if score > 0:
                scored.append((score, box))

        scored.sort(key=lambda x: -x[0])

        if scored:
            best_score = scored[0][0]
            # Only keep boxes within 60 % of the best score
            matched = [box for s, box in scored if s >= best_score * 0.6]
            # Cap at 2 boxes per field to keep the overlay readable
            field_boxes[field_name] = matched[:2]
        else:
            field_boxes[field_name] = []

    gov = ocr_result.get("government_warning", {})
    if gov.get("status") != "missing":
        gov_kw = {"government", "warning", "surgeon", "general", "pregnancy", "impairs"}
        matched = []
        for box in all_boxes:
            bt_tokens = set(box["text"].lower().split())
            if bt_tokens & gov_kw:
                matched.append(box)
        field_boxes["gov_warning"] = matched
    else:
        field_boxes["gov_warning"] = []

    return field_boxes


def _overall_confidence(result: dict) -> float:
    confs = []
    for f in result.get("fields", {}).values():
        c = f.get("confidence")
        if c is not None:
            confs.append(c)
    gov_conf = result.get("government_warning", {}).get("confidence")
    if gov_conf is not None:
        confs.append(gov_conf)
    return round(sum(confs) / len(confs), 3) if confs else 0.0


@app.route("/")
async def index():
    return await render_template("index.html")


@app.route("/healthz")
async def healthz():
    return {"status": "ok"}, 200


@app.route("/api/verify", methods=["POST"])
async def verify():
    t_start = time.perf_counter()

    form = await request.form
    files = (await request.files).getlist("files")

    brand_name = form.get("brand_name", "").strip()
    alcohol_content = form.get("alcohol_content", "").strip()
    net_contents = form.get("net_contents", "").strip()
    gov_warning_expected = form.get("gov_warning_expected", "false") == "true"
    debug = form.get("debug", "false") == "true"
    engine = form.get("engine", "tesseract")
    if engine not in ("paddleocr", "tesseract"):
        engine = "tesseract"

    log.info(
        "POST /api/verify  files=%d  engine=%s  brand=%r  alc=%s  net=%s  gov=%s  debug=%s",
        len(files), engine, brand_name, alcohol_content, net_contents, gov_warning_expected, debug,
    )

    results = []

    for f in files:
        t_file = time.perf_counter()
        # Save uploaded file to a temp path for OCR processing
        suffix = os.path.splitext(f.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name

        log.info("  [%s] Saved to %s, running OCR (%s)…", f.filename, tmp_path, engine)

        try:
            ocr_result = await asyncio.to_thread(
                ocr_service.evaluate_image,
                image_path=tmp_path,
                engine=engine,
                ttb_id=None,
                brand_name=brand_name or None,
                alcohol_content=alcohol_content or None,
                net_contents=net_contents or None,
                require_government_warning=gov_warning_expected,
                run_fallbacks=True,
            )
        except Exception:
            log.exception("  [%s] OCR failed", f.filename)
            raise
        finally:
            os.unlink(tmp_path)

        elapsed_file = round((time.perf_counter() - t_file) * 1000)
        fields_summary = {
            k: f"{v['status']} ({v.get('confidence', 0):.0%})"
            for k, v in ocr_result.get("fields", {}).items()
        }
        gov_status = ocr_result.get("government_warning", {}).get("status", "n/a")
        log.info(
            "  [%s] Done in %dms  fields=%s  gov_warning=%s  warnings=%s",
            f.filename, elapsed_file, fields_summary, gov_status,
            ocr_result.get("warnings", []),
        )

        entry = {
            "filename": f.filename,
            "engine": engine,
            "fields": {
                "brand_name": _translate_field(ocr_result["fields"]["brand_name"]),
                "alcohol_content": _translate_field(ocr_result["fields"]["alcohol_content"]),
                "net_contents": _translate_field(ocr_result["fields"]["net_contents"]),
                "gov_warning": _translate_gov_warning(
                    ocr_result["government_warning"], gov_warning_expected
                ),
            },
            "field_boxes": _associate_field_boxes(ocr_result),
            "overall_confidence": _overall_confidence(ocr_result),
            "summary": {
                "brand_name": ocr_result["fields"]["brand_name"]["status"],
                "alcohol_content": ocr_result["fields"]["alcohol_content"]["status"],
                "net_contents": ocr_result["fields"]["net_contents"]["status"],
                "government_warning": ocr_result["government_warning"]["status"],
            },
            "warnings": ocr_result.get("warnings", []),
        }

        if debug:
            entry["debug"] = {
                "raw_ocr_text": ocr_result.get("raw_ocr_text", ""),
                "normalized_ocr_text": ocr_result.get("normalized_ocr_text", ""),
                "ocr_by_angle": ocr_result.get("ocr_by_angle", {}),
                "ocr_attempts": ocr_result.get("ocr_attempts", []),
                "government_warning_detail": ocr_result.get("government_warning", {}),
                "fields_detail": ocr_result.get("fields", {}),
            }

        results.append(entry)

    elapsed_total = round((time.perf_counter() - t_start) * 1000)
    log.info("Verify complete  %d file(s) in %dms", len(results), elapsed_total)

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True)