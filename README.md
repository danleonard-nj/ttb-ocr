# TTB Label Verification

Automated OCR pipeline for verifying U.S. alcohol beverage labels against [TTB (Alcohol and Tobacco Tax and Trade Bureau)](https://www.ttb.gov/) regulatory data. Extracts and validates **brand name**, **alcohol content**, **net contents**, and **government warning** presence from label images.

## Architecture

```
treasury-takehome/
├── application/              # Web service (Quart + Bootstrap)
│   ├── app.py                # Async API server
│   ├── Dockerfile
│   ├── services/
│   │   ├── ocr.py            # OCR orchestrator + fallback logic
│   │   ├── matcher.py        # Field matching & gov warning detection
│   │   └── engines/
│   │       ├── base.py       # Abstract engine + shared image ops
│   │       ├── paddleocr.py  # PaddleOCR (PP-OCRv4) engine
│   │       └── tesseract.py  # Tesseract engine
│   ├── templates/            # Jinja2 HTML
│   └── static/               # CSS + JS
│
├── tools/                    # Reproducible engineering tools
│   ├── scrape_labels.py      # Data collection (scrape TTB, download images, distort)
│   └── benchmark_ocr.py      # Batch evaluation against ground truth
│
├── dataset/                  # Ground truth CSVs + raw label images
│   └── sample_labels/        # Small curated subset for quick demo runs
│
├── results/                  # Benchmark output (gitignored)
├── requirements.txt
└── README.md
```

## How It Works

### OCR Pipeline

1. **Image loading** — OpenCV reads the label image
2. **Primary pass** — Run OCR at 0° with basic preprocessing
3. **Field evaluation** — `LabelMatcher` checks extracted text against expected values
4. **Fallback passes** — If critical fields are missing, retry with rotations (90°/270°), region crops (right strip, bottom strip), and alternate preprocessing modes
5. **Merge & re-evaluate** — Combine all text, run final field matching
6. **Bounding box association** — Map detected text boxes back to matched fields

### Dual Engine Support

| Engine        | Strengths                               | Fallbacks                                     | Parallel              |
| ------------- | --------------------------------------- | --------------------------------------------- | --------------------- |
| **PaddleOCR** | Higher accuracy, handles varied layouts | 3 attempts (rotations + bottom crop)          | No (model serializes) |
| **Tesseract** | Lightweight, configurable preprocessing | 7 attempts (rotations + crops + filter modes) | Yes                   |

### Field Matching

- **Brand name** — Token overlap scoring (≥80% = partial, exact = found)
- **Alcohol content** — Regex extraction of `N%` patterns, float comparison (0.01 tolerance)
- **Net contents** — Regex extraction + unit normalization (ml, l, oz)
- **Government warning** — Multi-group pattern matching across 4 required segments (header, authority, pregnancy, impairment)

### Text Normalization

Built-in correction for common OCR misreads: `waming→warning`, `govemment→government`, `750mi→750ml`, `11 5%→11.5%`, etc.

## Quick Start

### Web Application

```bash
cd application
pip install -r requirements.txt
python app.py
```

Open `http://localhost:5000`. Upload label images, enter expected values, click **Verify Labels**.

### Docker

```bash
cd application
docker build -t ttb-verify .
docker run -p 8000:8000 ttb-verify
```

### Batch Evaluation

```bash
pip install -r requirements.txt
python tools/benchmark_ocr.py
```

Reads `dataset/master_filtered.csv`, runs OCR on each image, outputs to `results/`:

- `results_detail.csv` / `results_detail.json` — per-image results
- `results_summary.json` — aggregate accuracy stats

A small set of sample labels is included in `dataset/sample_labels/` so reviewers can run the benchmark immediately without fetching the full dataset.

### Data Collection

```bash
pip install -r requirements.txt
python tools/scrape_labels.py
```

Fetches TTB detail pages, downloads label images to `dataset/images/raw/`, builds `dataset/master.csv`.

The scraping utility demonstrates how the dataset could be automatically expanded from public TTB label records to support future model training or evaluation.

## V1 Results (PaddleOCR, n=5)

| Field              | Accuracy |
| ------------------ | -------- |
| Brand Name         | 100%     |
| Alcohol Content    | 100%     |
| Net Contents       | 80%      |
| Government Warning | 80%      |

## Dependencies

- **PaddleOCR** (PP-OCRv4) — primary OCR engine
- **Tesseract** — fallback OCR engine
- **Quart** — async web framework
- **OpenCV** — image processing
- **pandas** — data handling

## Web UI Features

- Drag-and-drop multi-image upload
- Per-field confidence scores with color-coded status badges
- Bounding box overlays on annotated images showing where each field was detected
- Debug mode with raw OCR text, attempt logs, and government warning detail
- Engine selector (PaddleOCR / Tesseract)
