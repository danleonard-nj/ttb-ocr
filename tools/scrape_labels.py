"""
TTB COLA Dataset Builder
Reads TTB CSV export, fetches detail pages + label images, writes results CSV.

Setup:
    pip install beautifulsoup4 curl_cffi

    Place TTB CSV export at: raw/ttb_export.csv
    Run:  python app.py
"""

import csv
import random
import re
import sys
import time
import urllib3
from pathlib import Path
from urllib.parse import urljoin

# Resolve project root so relative paths work regardless of CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup

urllib3.disable_warnings()

# ---------------------------------------------------------------------------
# Browser-like headers (Chrome 131 on Windows 10)
# ---------------------------------------------------------------------------
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CSV_PATH = PROJECT_ROOT / "ttb.csv"
IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
RAW_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images" / "raw"
OUTPUT_CSV = PROJECT_ROOT / "dataset" / "metadata.csv"

LIMIT = 300          # set to None to process all
DELAY = 3        # seconds between requests

# Debug / reprocessing options
# DEBUG_IDS = ['18108001000792', '11364001000181']       # If non-empty, ONLY process these TTB IDs, e.g. ["11364001000181"]
DEBUG_IDS = []
OVERWRITE = False    # If True, overwrite existing rows (lets you re-process)
SKIP_PARTIALS = False  # If True, skip image download for records missing core metadata
RUN_DISTORTIONS = True  # If True, run the distortion stage after scraping
REGEN_DISTORTIONS = False  # If True, delete existing distortions and regenerate from scratch

DISTORT_FRACTION = 0.5     # fraction of OK images to distort
VARIANTS_PER_IMAGE = 2     # distorted copies per selected image
BASE_URL = "https://ttbonline.gov"
MAX_RETRIES = 3
CAPTCHA_RETRIES = 3
CAPTCHA_BACKOFF = 30   # seconds, doubles each retry


# ---------------------------------------------------------------------------
# Distortion helpers
# ---------------------------------------------------------------------------

def _apply_gaussian_blur(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str]:
    k = rng.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0), f"blur_k{k}"


def _apply_rotation(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str]:
    angle = rng.uniform(1, 3) * rng.choice([-1, 1])
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return rotated, f"rotation_{angle:+.1f}deg"


def _apply_brightness(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str]:
    factor = round(rng.uniform(0.8, 1.2), 2)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), f"bright_{factor}"


def _apply_gaussian_noise(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str]:
    sigma = rng.randint(5, 15)
    noise = np.zeros(img.shape, dtype=np.float64)
    cv2.randn(noise, 0, sigma)  # deterministic enough with cv2
    noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy, f"noise_s{sigma}"


def _apply_perspective_warp(img: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str]:
    h, w = img.shape[:2]
    offset = rng.randint(5, 15)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [rng.randint(0, offset), rng.randint(0, offset)],
        [w - rng.randint(0, offset), rng.randint(0, offset)],
        [w - rng.randint(0, offset), h - rng.randint(0, offset)],
        [rng.randint(0, offset), h - rng.randint(0, offset)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return warped, f"persp_{offset}px"


_DISTORTIONS = [
    _apply_gaussian_blur,
    _apply_rotation,
    _apply_brightness,
    _apply_gaussian_noise,
    _apply_perspective_warp,
]


def distort_image(src_path: Path, ttb_id: str, rng: random.Random) -> list[dict]:
    """Generate VARIANTS_PER_IMAGE distorted copies; return metadata dicts."""
    img = cv2.imread(str(src_path))
    if img is None:
        return []

    variants: list[dict] = []
    for _ in range(VARIANTS_PER_IMAGE):
        # 1-2 distortions, biased toward 1
        n_distortions = 1 if rng.random() < 0.7 else 2
        chosen = rng.sample(_DISTORTIONS, n_distortions)

        result = img.copy()
        labels: list[str] = []
        for fn in chosen:
            result, label = fn(result, rng)
            labels.append(label)

        variant_label = "+".join(labels)
        dest_name = f"{ttb_id}_dist_{variant_label}.jpg"
        dest_path = IMAGES_DIR / dest_name
        cv2.imwrite(str(dest_path), result)

        variants.append({
            "image_path": f"images/{dest_name}",
            "variant_type": variant_label,
            "base_image": f"images/{src_path.name}",
        })
    return variants


# ---------------------------------------------------------------------------
# Distortion stage
# ---------------------------------------------------------------------------

def run_distortion_stage():
    """Read metadata.csv, distort a fraction of OK images, write master.csv."""
    rng = random.Random(42)

    # Read original metadata
    rows: list[dict] = []
    with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fields = list(reader.fieldnames) if reader.fieldnames else []
        for row in reader:
            rows.append(dict(row))

    # Load existing master.csv to find already-distorted TTB IDs
    master_csv = PROJECT_ROOT / "dataset" / "master.csv"
    existing_distorted_ids: set[str] = set()
    existing_master_rows: list[dict] = []
    master_fields = original_fields + ["variant_type", "base_image"]
    if REGEN_DISTORTIONS:
        # Wipe existing distortion images
        for p in IMAGES_DIR.glob("*_dist_*"):
            p.unlink()
        print("  Regenerating all distortions from scratch")
    elif master_csv.exists():
        with open(master_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_master_rows.append(dict(row))
                if row.get("variant_type", "original") != "original":
                    existing_master_rows_id = row.get("TTB ID", "").strip()
                    if existing_master_rows_id:
                        existing_distorted_ids.add(existing_master_rows_id)

    # Filter to images eligible for distortion
    eligible = [
        r for r in rows
        if r.get("fetch_status") == "ok" and r.get("image_path", "").strip()
    ]
    n_to_distort = max(1, int(len(eligible) * DISTORT_FRACTION))
    selected = rng.sample(eligible, min(n_to_distort, len(eligible)))

    # Skip IDs that already have variants
    to_process = [r for r in selected if r.get("TTB ID", "").strip() not in existing_distorted_ids]

    print(f"\n--- Distortion stage ---")
    print(f"Eligible images: {len(eligible)}, selected: {len(selected)}, already done: {len(selected) - len(to_process)}, to process: {len(to_process)}")

    # OK and OK_PARTIAL rows with images become originals in master.csv
    ok_rows = [r for r in rows if r.get("fetch_status") in ("ok", "ok_partial") and r.get("image_path", "").strip()]
    master_rows: list[dict] = []
    for r in ok_rows:
        r["variant_type"] = "original"
        r["base_image"] = ""
        master_rows.append(r)
    if existing_master_rows and existing_distorted_ids:
        existing_variant_rows = [r for r in existing_master_rows if r.get("variant_type", "original") != "original"]
        master_rows.extend(existing_variant_rows)

    # Generate distortions only for new IDs
    for i, rec in enumerate(to_process):
        ttb_id = rec.get("TTB ID", "").strip()
        img_rel = rec["image_path"].strip()
        src_path = PROJECT_ROOT / "dataset" / img_rel

        if not src_path.exists():
            print(f"  [{i+1}/{len(to_process)}] {ttb_id} – image missing, skipped")
            continue

        variants = distort_image(src_path, ttb_id, rng)
        print(f"  [{i+1}/{len(to_process)}] {ttb_id} – {len(variants)} variants")

        for v in variants:
            new_row = dict(rec)
            new_row["image_path"] = v["image_path"]
            new_row["variant_type"] = v["variant_type"]
            new_row["base_image"] = v["base_image"]
            master_rows.append(new_row)

    # Write master CSV
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=master_fields)
        writer.writeheader()
        writer.writerows(master_rows)

    n_distorted = sum(1 for r in master_rows if r["variant_type"] != "original")
    print(f"Master CSV -> {master_csv}  ({len(master_rows)} rows, {n_distorted} distorted)")


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------
def extract_all_fields(soup: BeautifulSoup) -> dict[str, str]:
    """Dynamically extract every label→data pair from a TTB detail page.

    Works across all known form versions without hard-coded field numbers.
    """
    fields: dict[str, str] = {}

    for label_div in soup.find_all(
        "div",
        class_=lambda c: c and ("label" in c or "boldlabel" in c),
    ):
        raw_label = label_div.get_text(" ", strip=True)
        # Normalize: strip "6." / "8a." prefix, parenthesized hints
        clean = re.sub(r"^\d+[a-z]?\.\s*", "", raw_label).strip()
        clean = re.sub(r"\s*\(.*?\)\s*", " ", clean).strip()
        clean = re.sub(r"\s+", " ", clean).upper().strip()
        if not clean:
            continue

        # Find the data div that belongs to *this* label.
        # Prefer the next-sibling data div (handles shared <td>s like
        # QUALIFICATIONS / STATUS / CLASS/TYPE that live in the same cell).
        data_text = ""
        for sib in label_div.find_next_siblings("div", class_="data"):
            t = sib.get_text(" ", strip=True)
            if t:
                data_text = t
                break

        if not data_text:
            td = label_div.find_parent("td")
            if td:
                for child in td.find_all("div", class_="data", recursive=False):
                    t = child.get_text(" ", strip=True)
                    if t:
                        data_text = t
                        break

        if data_text:
            fields[clean] = data_text

    return fields


def normalize_net_contents(text: str | None) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*"
        r"(milliliters?|ml|liters?|l|fl\.?\s*oz|ounces?|gal\.?|gallons?)",
        t,
    )
    if not m:
        return text.strip()
    qty = m.group(1)
    unit = m.group(2)
    if "millil" in unit or unit == "ml":
        return f"{qty}ml"
    if "liter" in unit or unit == "l":
        return f"{qty}l"
    if "oz" in unit or "ounce" in unit:
        return f"{qty}oz"
    if "gal" in unit:
        return text.strip()  # keep full multi-size string for kegs
    return text.strip()


def normalize_abv(text: str | None) -> str:
    if not text:
        return ""
    m = re.search(r"\d+(?:\.\d+)?", text)
    return m.group(0) if m else text.strip()


def is_captcha(text: str) -> bool:
    return "What code is in the image" in text or "are you a human" in text.lower()


# ---------------------------------------------------------------------------
# Fetch one record
# ---------------------------------------------------------------------------
def fetch_record(session: cffi_requests.Session, ttb_id: str) -> dict:
    url = (f"{BASE_URL}/colasonline/viewColaDetails.do"
           f"?action=publicFormDisplay&ttbid={ttb_id}")

    result = {
        "brand_name_detail": "",
        "fanciful_name_detail": "",
        "net_contents": "",
        "alcohol_content": "",
        "image_path": "",
        "image_count": "0",
        "fetch_status": "",
    }

    headers = {
        "Referer": f"{BASE_URL}/colasonline/publicSearchColasBasic.do",
    }

    try:
        r = session.get(url, headers=headers, verify=False, timeout=30)
        r.raise_for_status()
    except Exception as e:
        result["fetch_status"] = f"error: {e}"
        return result

    # Retry with backoff on captcha
    if is_captcha(r.text):
        for attempt in range(CAPTCHA_RETRIES):
            wait = CAPTCHA_BACKOFF * (2 ** attempt)
            print(f"captcha! waiting {wait}s (retry {attempt+1}/{CAPTCHA_RETRIES})...", end=" ", flush=True)
            time.sleep(wait)
            try:
                r = session.get(url, headers=headers, verify=False, timeout=30)
                r.raise_for_status()
            except Exception as e:
                result["fetch_status"] = f"error: {e}"
                return result
            if not is_captcha(r.text):
                break
        else:
            result["fetch_status"] = "captcha"
            return result

    soup = BeautifulSoup(r.text, "html.parser")
    page_fields = extract_all_fields(soup)

    result["brand_name_detail"] = page_fields.get("BRAND NAME", "")
    result["fanciful_name_detail"] = page_fields.get("FANCIFUL NAME", "")
    result["net_contents"] = normalize_net_contents(page_fields.get("NET CONTENTS"))
    result["alcohol_content"] = normalize_abv(page_fields.get("ALCOHOL CONTENT"))

    # Early exit if metadata is incomplete and we're skipping partials
    core_fields = [
        result["brand_name_detail"],
        result["fanciful_name_detail"],
        result["net_contents"],
        result["alcohol_content"],
    ]
    if SKIP_PARTIALS and not all(core_fields):
        result["fetch_status"] = "ok_partial"
        return result

    # Find ALL label images on the page
    imgs = soup.find_all("img", src=lambda x: x and "publicViewAttachment" in x)
    if not imgs:
        result["fetch_status"] = "no_image"
        return result

    if len(imgs) > 1:
        print(f"  !!!! Warning: found {len(imgs)} images for {ttb_id}, will combine into one")

    ext_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/tiff": ".tif",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/webp": ".webp",
    }

    raw_paths: list[Path] = []
    try:
        for idx, img in enumerate(imgs, start=1):
            img_url = urljoin(f"{BASE_URL}/", img["src"]).replace(" ", "%20")
            img_headers = {
                "Referer": url,
                "Sec-Fetch-Dest": "image",
                "Sec-Fetch-Mode": "no-cors",
            }
            img_r = session.get(img_url, headers=img_headers, verify=False, timeout=30)
            img_r.raise_for_status()

            content_type = img_r.headers.get("Content-Type", "")
            ext = ext_map.get(content_type.split(";")[0].strip().lower())
            if not ext:
                print(f"Warning: Unknown Content-Type '{content_type}' for {img_url}, guessing extension from URL")
                ext = Path(img_url.split("?")[0]).suffix.lower() or ".jpg"

            # Save each image separately in raw/ for debugging
            raw_dest = RAW_IMAGES_DIR / f"{ttb_id}_{idx}{ext}"
            with open(raw_dest, "wb") as f:
                f.write(img_r.content)
            raw_paths.append(raw_dest)

        # Combine all images into one vertically-stacked composite
        pil_images = []
        for p in raw_paths:
            pil_img = Image.open(p)
            pil_images.append(pil_img)

        if len(pil_images) == 1:
            combined = pil_images[0]
        else:
            total_height = sum(im.height for im in pil_images)
            max_width = max(im.width for im in pil_images)
            combined = Image.new("RGB", (max_width, total_height), (255, 255, 255))
            y_offset = 0
            for im in pil_images:
                combined.paste(im, (0, y_offset))
                y_offset += im.height

        combined_dest = IMAGES_DIR / f"{ttb_id}.jpg"
        combined.save(combined_dest)

        result["image_path"] = f"images/{ttb_id}.jpg"
        result["image_count"] = str(len(pil_images))

        # ok = all 4 core metadata fields present; ok_partial = at least one missing
        core_fields = [
            result["brand_name_detail"],
            result["fanciful_name_detail"],
            result["net_contents"],
            result["alcohol_content"],
        ]
        if all(core_fields):
            result["fetch_status"] = "ok"
        else:
            result["fetch_status"] = "ok_partial"

        if len(imgs) > 1:
            print(f"({len(imgs)} images) ", end="")
    except Exception as e:
        result["fetch_status"] = f"image_error: {e}"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found")
        sys.exit(1)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Load CSV
    records = []
    with open(CSV_PATH, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip() for h in reader.fieldnames]
        for row in reader:
            row.pop(None, None)  # drop extra columns beyond header
            row["TTB ID"] = row["TTB ID"].strip().strip("'\"")
            records.append(dict(row))

    print(f"Loaded {len(records)} records")
    if DEBUG_IDS:
        debug_set = set(DEBUG_IDS)
        records = [r for r in records if r["TTB ID"] in debug_set]
        print(f"Debug mode: filtered to {len(records)} IDs: {DEBUG_IDS}")
    elif LIMIT:
        records = records[:LIMIT]
        print(f"Processing first {LIMIT}")

    # curl_cffi session impersonates Chrome's TLS fingerprint (JA3/JA4)
    session = cffi_requests.Session(impersonate="chrome131")
    session.headers.update(BROWSER_HEADERS)

    # Warm up: visit the search page first to establish cookies
    print("Warming up session (visiting search page)...")
    try:
        warmup = session.get(
            f"{BASE_URL}/colasonline/publicSearchColasBasic.do",
            verify=False, timeout=30,
        )
        print(f"  warmup status={warmup.status_code}, cookies={len(session.cookies)}")
    except Exception as e:
        print(f"  warmup failed: {e} (continuing anyway)")

    extra_cols = ["brand_name_detail", "fanciful_name_detail",
                  "net_contents", "alcohol_content", "image_path", "image_count", "fetch_status"]
    all_cols = list(records[0].keys()) + extra_cols

    # Resume: load already-processed TTB IDs from existing output
    # Deduplicate by TTB ID (last row per ID wins) so retried failures
    # don't accumulate duplicate rows.
    done_ids = set()
    existing_by_id: dict[str, dict] = {}
    # When SKIP_PARTIALS is off, ok_partial records need re-processing
    # (to download images that were skipped before), so exclude them from done_ids.
    done_statuses = {"ok", "ok_partial"} if SKIP_PARTIALS else {"ok"}
    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = row.get("TTB ID", "").strip()
                existing_by_id[tid] = row
                if row.get("fetch_status") in done_statuses:
                    done_ids.add(tid)
        print(f"Resuming: {len(done_ids)} already completed, {len(existing_by_id)} unique in CSV")
    existing_rows = list(existing_by_id.values())

    # OVERWRITE: drop target IDs from existing rows so they get re-processed
    overwrite_ids = set()
    if OVERWRITE and OUTPUT_CSV.exists():
        if DEBUG_IDS:
            overwrite_ids = set(DEBUG_IDS)
        else:
            overwrite_ids = {r["TTB ID"] for r in records}
        existing_rows = [r for r in existing_rows if r.get("TTB ID", "").strip() not in overwrite_ids]
        done_ids -= overwrite_ids
        print(f"Overwrite mode: will re-process {len(overwrite_ids)} IDs")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # IDs that will be (re-)processed in the loop below
    will_process = {r["TTB ID"] for r in records} - done_ids
    # Only write rows we will NOT touch again — avoids duplicates
    kept_rows = [r for r in existing_rows
                 if r.get("TTB ID", "").strip() not in will_process]

    # Always rewrite with deduplicated rows, then append new ones
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for row in kept_rows:
            writer.writerow(row)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)

        for i, rec in enumerate(records):
            ttb_id = rec["TTB ID"]
            brand = rec.get("Brand Name", "")

            if ttb_id in done_ids:
                print(f"[{i+1}/{len(records)}] {ttb_id} {brand}... skipped (already done)")
                continue

            # If image already on disk, use existing metadata from CSV — no web request
            combined_path = IMAGES_DIR / f"{ttb_id}.jpg"
            prev = existing_by_id.get(ttb_id)
            if combined_path.exists() and prev:
                rec.update(prev)
                print(f"[{i+1}/{len(records)}] {ttb_id} {brand}... (image exists, kept from CSV)")
                writer.writerow(rec)
                f.flush()
                continue

            print(f"[{i+1}/{len(records)}] {ttb_id} {brand}...", end=" ", flush=True)

            detail = fetch_record(session, ttb_id)
            status = detail["fetch_status"]
            print(status)

            rec.update(detail)
            writer.writerow(rec)
            f.flush()

            time.sleep(max(0, DELAY + random.uniform(-1, 1)))

    print(f"\nDone! -> {OUTPUT_CSV}")
    print(f"Images -> {IMAGES_DIR}/ ({len(list(IMAGES_DIR.glob('*')))} files)")

    # --- Distortion stage ---
    if RUN_DISTORTIONS:
        run_distortion_stage()


if __name__ == "__main__":
    main()
