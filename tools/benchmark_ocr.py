"""
OCR Comparison Pipeline
Loads master_filtered.csv, OCRs each image, compares extracted fields
against ground truth, and generates detailed + summary reports.

Usage:
    python compare.py
"""

import csv
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

# Resolve project root so imports and relative paths work
# regardless of the working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from application.services.ocr import evaluate_image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET_DIR = PROJECT_ROOT / "dataset"
MASTER_CSV = DATASET_DIR / "master_filtered.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results_detail.csv"
RESULTS_JSON = RESULTS_DIR / "results_detail.json"
SUMMARY_JSON = RESULTS_DIR / "results_summary.json"

OCR_ENGINE = "paddleocr"  # "tesseract" or "paddleocr"

FIELD_NAMES = ["brand_name", "alcohol_content", "net_contents"]

MAX_PROCESSED = 5  # Set to an integer to limit number of records processed (for testing)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_records(csv_path: Path) -> list[dict]:
    """Load master CSV and return list of record dicts."""
    df = pd.read_csv(csv_path, dtype=str)
    df = df.fillna("")
    return df.to_dict(orient="records")


def _default_missing_fields() -> dict:
    return {
        f: {
            "raw_value": None,
            "value": None,
            "normalized_value": None,
            "confidence": 0.0,
            "status": "missing",
        }
        for f in FIELD_NAMES
    }


def _default_missing_gov_warning() -> dict:
    return {
        "status": "missing",
        "confidence": 0.0,
        "matched_groups": {},
        "matched_anchors": [],
    }


def _safe_mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return round(statistics.median(values), 2) if values else 0.0


def _print_record_debug(rec: dict, result: dict):
    """Print detailed per-record debug output."""
    print("    --- expected ---")
    print(f"    brand_name      : {rec.get('Brand Name', '')}")
    print(f"    alcohol_content : {rec.get('alcohol_content', '')}")
    print(f"    net_contents    : {rec.get('net_contents', '')}")
    print(f"    image_path      : {result.get('image_path', '')}")

    print("    --- OCR raw ---")
    raw = result.get("ocr_text", "") or ""
    print(f"    {raw if raw else '[empty]'}")

    print("    --- OCR by angle ---")
    ocr_by_angle = result.get("ocr_by_angle", {}) or {}
    for angle in (0, 90, 270):
        text = ocr_by_angle.get(angle, "") or "[empty]"
        print(f"    angle {angle:<3}: {text}")

    print("    --- normalized fields ---")
    for fname in FIELD_NAMES:
        fdata = result["fields"].get(fname, {})
        print(
            f"    {fname:<16} "
            f"status={fdata.get('status', ''):<8} "
            f"conf={fdata.get('confidence', '')!s:<6} "
            f"expected={fdata.get('normalized_value', '')!r} "
            f"found={fdata.get('raw_value', '')!r}"
        )

    gov = result.get("government_warning", {})
    print("    --- government warning ---")
    print(
        f"    status={gov.get('status', '')} "
        f"conf={gov.get('confidence', '')} "
        f"anchors={gov.get('matched_anchors', [])}"
    )
    print(f"    groups={json.dumps(gov.get('matched_groups', {}), ensure_ascii=False)}")

    print("    --- warnings / metrics ---")
    print(f"    warnings        : {result.get('warnings', [])}")
    print(f"    ocr_engine      : {result.get('metrics', {}).get('ocr_engine', '')}")
    print(f"    ocr_time_ms     : {result.get('metrics', {}).get('ocr_time_ms', 0.0)}")
    print(f"    total_time_ms   : {result.get('metrics', {}).get('total_time_ms', 0.0)}")
    print()


def process_record(rec: dict) -> dict:
    """OCR one image and evaluate against ground truth."""
    record_t0 = time.perf_counter()

    ttb_id = rec.get("TTB ID", "").strip()
    image_rel = rec.get("image_path", "").strip()
    image_path = str(DATASET_DIR / image_rel) if image_rel else ""

    brand_name = rec.get("Brand Name", "") or None
    alcohol_content = rec.get("alcohol_content", "") or None
    net_contents = rec.get("net_contents", "") or None

    if not image_rel or not Path(image_path).exists():
        total_time_ms = round((time.perf_counter() - record_t0) * 1000, 2)
        return {
            "ttb_id": ttb_id,
            "image_path": image_rel,
            "variant_type": rec.get("variant_type", ""),
            "ocr_text": "",
            "ocr_by_angle": {},
            "error": "image_not_found",
            "fields": _default_missing_fields(),
            "government_warning": _default_missing_gov_warning(),
            "warnings": ["image_not_found"],
            "metrics": {
                "ocr_engine": OCR_ENGINE,
                "ocr_time_ms": 0.0,
                "total_time_ms": total_time_ms,
            },
        }

    ocr_t0 = time.perf_counter()
    result = evaluate_image(
        image_path=image_path,
        engine=OCR_ENGINE,
        ttb_id=ttb_id,
        brand_name=brand_name,
        alcohol_content=alcohol_content,
        net_contents=net_contents,
        require_government_warning=True,
    )
    ocr_time_ms = round((time.perf_counter() - ocr_t0) * 1000, 2)
    total_time_ms = round((time.perf_counter() - record_t0) * 1000, 2)

    return {
        "ttb_id": ttb_id,
        "image_path": image_rel,
        "variant_type": rec.get("variant_type", ""),
        "ocr_text": result["raw_ocr_text"],
        "ocr_by_angle": result.get("ocr_by_angle", {}),
        "normalized_ocr_text": result.get("normalized_ocr_text", ""),
        "error": None,
        "fields": result["fields"],
        "government_warning": result["government_warning"],
        "warnings": result["warnings"],
        "metrics": {
            "ocr_engine": OCR_ENGINE,
            "ocr_time_ms": ocr_time_ms,
            "total_time_ms": total_time_ms,
        },
    }


def build_summary(results: list[dict], elapsed_seconds: float) -> dict:
    """Compute aggregate stats from the per-record results."""
    total = len(results)
    errors = sum(1 for r in results if r["error"])

    field_stats = {}
    for fname in FIELD_NAMES:
        counts = Counter()
        for r in results:
            status = r["fields"].get(fname, {}).get("status", "missing")
            counts[status] += 1

        field_stats[fname] = {
            "total": total,
            "found": counts.get("found", 0),
            "partial": counts.get("partial", 0),
            "missing": counts.get("missing", 0),
            "accuracy": round(counts.get("found", 0) / total, 4) if total else 0,
            "partial_rate": round(counts.get("partial", 0) / total, 4) if total else 0,
        }

    gov_counts = Counter()
    for r in results:
        status = r.get("government_warning", {}).get("status", "missing")
        gov_counts[status] += 1

    government_warning_stats = {
        "total": total,
        "found": gov_counts.get("found", 0),
        "partial": gov_counts.get("partial", 0),
        "missing": gov_counts.get("missing", 0),
        "accuracy": round(gov_counts.get("found", 0) / total, 4) if total else 0,
        "partial_rate": round(gov_counts.get("partial", 0) / total, 4) if total else 0,
    }

    total_times = [r.get("metrics", {}).get("total_time_ms", 0.0) for r in results]
    ocr_times = [r.get("metrics", {}).get("ocr_time_ms", 0.0) for r in results]

    timing_stats = {
        "engine": OCR_ENGINE,
        "total_elapsed_seconds": round(elapsed_seconds, 2),
        "records_per_second": round(total / elapsed_seconds, 3) if elapsed_seconds > 0 else 0.0,
        "avg_total_time_ms": _safe_mean(total_times),
        "median_total_time_ms": _safe_median(total_times),
        "min_total_time_ms": round(min(total_times), 2) if total_times else 0.0,
        "max_total_time_ms": round(max(total_times), 2) if total_times else 0.0,
        "avg_ocr_time_ms": _safe_mean(ocr_times),
        "median_ocr_time_ms": _safe_median(ocr_times),
        "min_ocr_time_ms": round(min(ocr_times), 2) if ocr_times else 0.0,
        "max_ocr_time_ms": round(max(ocr_times), 2) if ocr_times else 0.0,
    }

    variant_groups: dict[str, list[dict]] = {}
    for r in results:
        vt = r["variant_type"] or "unknown"
        variant_groups.setdefault(vt, []).append(r)

    variant_stats = {}
    for vt, group in variant_groups.items():
        n = len(group)
        vf = {}

        for fname in FIELD_NAMES:
            found = sum(1 for r in group if r["fields"].get(fname, {}).get("status") == "found")
            partial = sum(1 for r in group if r["fields"].get(fname, {}).get("status") == "partial")
            vf[fname] = {
                "found_rate": round(found / n, 4) if n else 0,
                "partial_rate": round(partial / n, 4) if n else 0,
            }

        gov_found = sum(1 for r in group if r.get("government_warning", {}).get("status") == "found")
        gov_partial = sum(1 for r in group if r.get("government_warning", {}).get("status") == "partial")

        group_total_times = [r.get("metrics", {}).get("total_time_ms", 0.0) for r in group]
        group_ocr_times = [r.get("metrics", {}).get("ocr_time_ms", 0.0) for r in group]

        variant_stats[vt] = {
            "count": n,
            "accuracy_by_field": vf,
            "government_warning": {
                "found_rate": round(gov_found / n, 4) if n else 0,
                "partial_rate": round(gov_partial / n, 4) if n else 0,
            },
            "timing": {
                "avg_total_time_ms": _safe_mean(group_total_times),
                "avg_ocr_time_ms": _safe_mean(group_ocr_times),
            },
        }

    all_warnings: list[str] = []
    for r in results:
        all_warnings.extend(r["warnings"])
    warning_counts = Counter(all_warnings).most_common(20)

    return {
        "total_records": total,
        "image_errors": errors,
        "field_stats": field_stats,
        "government_warning_stats": government_warning_stats,
        "timing_stats": timing_stats,
        "variant_stats": variant_stats,
        "top_warnings": warning_counts,
    }


def write_detail_csv(results: list[dict], path: Path):
    """Write one row per record with flattened field results."""
    fieldnames = [
        "ttb_id",
        "image_path",
        "variant_type",
        "error",
        "ocr_engine",
        "ocr_time_ms",
        "total_time_ms",
        "ocr_text",
        "normalized_ocr_text",
        "ocr_angle_0",
        "ocr_angle_90",
        "ocr_angle_270",
    ]

    for fname in FIELD_NAMES:
        fieldnames.extend([
            f"{fname}_status",
            f"{fname}_confidence",
            f"{fname}_expected",
            f"{fname}_found",
        ])

    fieldnames.extend([
        "government_warning_status",
        "government_warning_confidence",
        "government_warning_matched_anchors",
        "government_warning_matched_groups",
        "warnings",
    ])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            metrics = r.get("metrics", {})
            row = {
                "ttb_id": r["ttb_id"],
                "image_path": r["image_path"],
                "variant_type": r["variant_type"],
                "error": r["error"] or "",
                "ocr_engine": metrics.get("ocr_engine", ""),
                "ocr_time_ms": metrics.get("ocr_time_ms", ""),
                "total_time_ms": metrics.get("total_time_ms", ""),
                "ocr_text": r.get("ocr_text", ""),
                "normalized_ocr_text": r.get("normalized_ocr_text", ""),
                "ocr_angle_0": r.get("ocr_by_angle", {}).get(0, ""),
                "ocr_angle_90": r.get("ocr_by_angle", {}).get(90, ""),
                "ocr_angle_270": r.get("ocr_by_angle", {}).get(270, ""),
            }

            for fname in FIELD_NAMES:
                fdata = r["fields"].get(fname, {})
                row[f"{fname}_status"] = fdata.get("status", "")
                row[f"{fname}_confidence"] = fdata.get("confidence", "")
                row[f"{fname}_expected"] = fdata.get("normalized_value", "")
                row[f"{fname}_found"] = fdata.get("raw_value", "")

            gov = r.get("government_warning", {})
            row["government_warning_status"] = gov.get("status", "")
            row["government_warning_confidence"] = gov.get("confidence", "")
            row["government_warning_matched_anchors"] = "; ".join(gov.get("matched_anchors", []))
            row["government_warning_matched_groups"] = json.dumps(gov.get("matched_groups", {}), ensure_ascii=False)
            row["warnings"] = "; ".join(r["warnings"])

            writer.writerow(row)


def print_summary(summary: dict):
    """Print a human-readable summary to stdout."""
    print("=" * 72)
    print("  OCR COMPARISON RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Total records evaluated : {summary['total_records']}")
    print(f"  Image errors (missing)  : {summary['image_errors']}")
    print()

    print("  Timing:")
    ts = summary["timing_stats"]
    print(f"    Engine               : {ts['engine']}")
    print(f"    Total elapsed        : {ts['total_elapsed_seconds']:.2f}s")
    print(f"    Throughput           : {ts['records_per_second']:.3f} records/sec")
    print(f"    Avg total / record   : {ts['avg_total_time_ms']:.2f} ms")
    print(f"    Median total / record: {ts['median_total_time_ms']:.2f} ms")
    print(f"    Min total / record   : {ts['min_total_time_ms']:.2f} ms")
    print(f"    Max total / record   : {ts['max_total_time_ms']:.2f} ms")
    print(f"    Avg OCR / record     : {ts['avg_ocr_time_ms']:.2f} ms")
    print(f"    Median OCR / record  : {ts['median_ocr_time_ms']:.2f} ms")
    print()

    print("  Field-level accuracy:")
    print(f"  {'Field':<24} {'Found':>7} {'Partial':>8} {'Missing':>8} {'Accuracy':>9}")
    print("  " + "-" * 64)
    for fname in FIELD_NAMES:
        fs = summary["field_stats"][fname]
        print(
            f"  {fname:<24} {fs['found']:>7} {fs['partial']:>8} "
            f"{fs['missing']:>8} {fs['accuracy']:>8.1%}"
        )

    print()
    print("  Government warning:")
    gws = summary["government_warning_stats"]
    print(
        f"  {'government_warning':<24} {gws['found']:>7} {gws['partial']:>8} "
        f"{gws['missing']:>8} {gws['accuracy']:>8.1%}"
    )
    print()

    if summary["variant_stats"]:
        print("  Accuracy by variant type:")
        for vt, vs in summary["variant_stats"].items():
            field_parts = []
            for f, stats in vs["accuracy_by_field"].items():
                field_parts.append(f"{f}={stats['found_rate']:.0%}")

            field_parts.append(f"gov_warning={vs['government_warning']['found_rate']:.0%}")
            field_parts.append(f"avg_ms={vs['timing']['avg_total_time_ms']:.1f}")
            accs = ", ".join(field_parts)
            print(f"    {vt} (n={vs['count']}): {accs}")
        print()

    if summary["top_warnings"]:
        print("  Top warnings:")
        for warning, count in summary["top_warnings"]:
            print(f"    {warning}: {count}")

    print("=" * 72)


def main():
    global OCR_ENGINE

    import argparse
    parser = argparse.ArgumentParser(description="OCR Comparison Pipeline")
    parser.add_argument(
        "--engine",
        choices=["tesseract", "paddleocr"],
        default=OCR_ENGINE,
        help="OCR engine to use (default: tesseract)",
    )
    args = parser.parse_args()
    OCR_ENGINE = args.engine

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading records from {MASTER_CSV} ...")
    print(f"  OCR engine: {OCR_ENGINE}")
    records = load_records(MASTER_CSV)
    print(f"  {len(records)} records loaded.\n")

    results: list[dict] = []
    t0 = time.perf_counter()

    total_to_process = len(records) if MAX_PROCESSED is None else min(len(records), MAX_PROCESSED)

    for i, rec in enumerate(records):
        if MAX_PROCESSED is not None and i >= MAX_PROCESSED:
            break

        ttb_id = rec.get("TTB ID", "?")
        print(f"[{i + 1}/{total_to_process}] {ttb_id}")

        result = process_record(rec)

        statuses = [result["fields"][f]["status"] for f in FIELD_NAMES]
        status_parts = [f"{f}={s}" for f, s in zip(FIELD_NAMES, statuses)]
        status_parts.append(f"gov_warning={result.get('government_warning', {}).get('status', 'missing')}")
        status_parts.append(f"time_ms={result.get('metrics', {}).get('total_time_ms', 0.0):.2f}")
        print(f"    summary         : {', '.join(status_parts)}")

        _print_record_debug(rec, result)

        results.append(result)

    elapsed = time.perf_counter() - t0
    print(f"\nProcessed {len(results)} records in {elapsed:.1f}s\n")

    write_detail_csv(results, RESULTS_CSV)
    print(f"Detail CSV   -> {RESULTS_CSV}")

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detail JSON  -> {RESULTS_JSON}")

    summary = build_summary(results, elapsed_seconds=elapsed)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON -> {SUMMARY_JSON}\n")

    print_summary(summary)


if __name__ == "__main__":
    main()