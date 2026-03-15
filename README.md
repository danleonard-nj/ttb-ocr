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
├── k8s/                      # Kubernetes deployment
│   ├── treasury-assignment/  # Helm chart
│   └── ingress.yml           # Ingress configuration
│
├── .github/                  # CI/CD
│   └── workflows/            # GitHub Actions pipeline
│
├── tools/                    # Reproducible engineering tools
│   ├── scrape_labels.py      # Data collection from public TTB records
│   └── benchmark_ocr.py      # Batch evaluation against ground truth
│
├── dataset/                  # Ground truth CSVs + label images
│   └── sample_labels/        # Curated subset for quick demo runs
│
├── results/                  # Benchmark output (gitignored)
├── requirements.txt
├── V2_ROADMAP.md
└── README.md
```

## Approach

### Design Philosophy

The stakeholder interviews made one thing clear: this is a **matching** problem, not a comprehension problem. Agents need to verify that label content matches application data — fast, reliably, and with honest confidence signals when the system is uncertain.

Rather than building a heavyweight ML pipeline that tries to semantically parse entire labels, this system uses a **pattern-first extraction** approach: OCR the image, then search the resulting text for known field formats using regex, token matching, and targeted heuristics. This is faster, more deterministic, more debuggable, and doesn't hallucinate — all properties that matter in a federal compliance context.

The "AI" in this system is the OCR itself. Everything downstream is a rules engine, and that's the right tool for the job.

### Why This Architecture

- **5-second target** — Local OCR with deterministic matching runs in ~5 seconds per label. No network round-trips to cloud ML endpoints, no firewall issues.
- **No external API dependencies** — Runs entirely self-contained in a Docker container. No API keys, no outbound network required.
- **Explainable results** — Every match decision traces back to a regex pattern or token overlap score. Agents can see _why_ a field matched or didn't.
- **Batch support** — Multi-image upload processes labels concurrently.

### OCR Pipeline

1. **Image loading** — OpenCV reads the label image
2. **Primary pass** — OCR at 0° with basic preprocessing
3. **Field evaluation** — `LabelMatcher` checks extracted text against expected values
4. **Fallback passes** — If critical fields are missing, retry with rotations (90°/270°), region crops (right strip, bottom strip), and alternate preprocessing modes
5. **Merge & re-evaluate** — Combine all text, run final field matching
6. **Bounding box association** — Map detected text boxes back to matched fields

### Dual Engine Support

| Engine                   | Role                                                      | Fallback Strategy                            |
| ------------------------ | --------------------------------------------------------- | -------------------------------------------- |
| **Tesseract**            | Primary engine — lightweight, configurable preprocessing  | 7 attempts: rotations + crops + filter modes |
| **PaddleOCR** (PP-OCRv4) | Experimental — included to demonstrate engine abstraction | 3 attempts: rotations + bottom crop          |

Tesseract is the default engine. PaddleOCR is accessible via the debug toggle in the UI and is included to demonstrate the pluggable engine architecture — it needs further tuning before it's production-ready.

### Field Matching

| Field                  | Method                                | Match Logic                                                                                                                     |
| ---------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Brand name**         | Token overlap scoring                 | ≥80% overlap = partial match, exact = full match. Case-insensitive. Handles Dave's `STONE'S THROW` vs `Stone's Throw` scenario. |
| **Alcohol content**    | Regex extraction of `N%` patterns     | Float comparison with 0.01 tolerance                                                                                            |
| **Net contents**       | Regex extraction + unit normalization | Normalizes ml/mL/ML, l/L, oz/fl. oz. variants                                                                                   |
| **Government warning** | Multi-group pattern matching          | Checks 4 required segments: header, surgeon general authority, pregnancy clause, impairment clause                              |

### Text Normalization

Built-in correction for common OCR misreads: `waming→warning`, `govemment→government`, `750mi→750ml`, `11 5%→11.5%`, etc. These were identified empirically during benchmark evaluation against the test dataset.

## Live Demos

| Deployment      | URL                                                                                 | Infrastructure                          |
| --------------- | ----------------------------------------------------------------------------------- | --------------------------------------- |
| **AKS**         | [treasury-assignment.dan-leonard.com](https://treasury-assignment.dan-leonard.com/) | Azure Kubernetes Service via Helm       |
| **App Service** | [treasury-test-app.azurewebsites.net](https://treasury-test-app.azurewebsites.net/) | Azure App Service B1 via GitHub Actions |

Both deployments run the same Docker image from ACR. Two deployment paths are included to demonstrate both a CI/CD pipeline (GitHub Actions → App Service) and infrastructure-as-code (Helm chart → AKS).

**Deployment note:** The application was initially deployed to an App Service B1 tier, which introduced significant latency on OCR workloads (~15s per label) due to the single-core sandbox environment. Rather than immediately scaling up the App Service plan, the same image was deployed to an existing AKS cluster with configurable resource limits to isolate whether the bottleneck was application-level or infrastructure-level. This confirmed the issue was compute-bound:

| Environment      | Response Time | Notes                                        |
| ---------------- | ------------- | -------------------------------------------- |
| App Service B1   | ~15s          | Single-core sandbox, budget tier             |
| AKS              | ~5s           | Pod resource limits closer to dev machine    |
| App Service P0V3 | ~3s           | Lowest end of production-grade compute tiers |

The App Service is currently on B1 for cost reasons; the AKS deployment is the recommended demo environment. For a production deployment, right-sizing the compute tier to the OCR workload profile would be part of capacity planning.

## Quick Start

### Web Application (Local)

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

### Kubernetes

Helm chart and ingress configuration are in `k8s/`.

```bash
# Build and push to ACR
docker build -t <your-acr>.azurecr.io/ttb-verify:latest application/
docker push <your-acr>.azurecr.io/ttb-verify:latest

# Deploy via Helm
helm install treasury-assignment k8s/treasury-assignment/
kubectl apply -f k8s/ingress.yml
```

### Batch Evaluation

```bash
pip install -r requirements.txt
python tools/benchmark_ocr.py
```

Reads `dataset/master_filtered.csv`, runs OCR on each image, outputs per-image results and aggregate accuracy stats to `results/`. A small set of sample labels is included in `dataset/sample_labels/` so reviewers can run the benchmark immediately without fetching the full dataset.

### Data Collection

```bash
python tools/scrape_labels.py
```

Fetches public TTB label detail pages and downloads label images to `dataset/images/raw/`. This demonstrates how the dataset could be automatically expanded from public TTB records for future evaluation or model training.

## Results (Tesseract, n=5)

| Field              | Accuracy |
| ------------------ | -------- |
| Brand Name         | 100%     |
| Alcohol Content    | 100%     |
| Net Contents       | 80%      |
| Government Warning | 80%      |

Sample size is small — these numbers demonstrate the pipeline works, not production-grade accuracy. See `V2_ROADMAP.md` for scaling plans.

## Web UI

- Drag-and-drop multi-image upload with batch support
- Per-field confidence scores with color-coded status badges (green/yellow/red)
- Bounding box overlays on annotated images showing where each field was detected
- Debug mode with raw OCR text, attempt logs, government warning detail, and alternate engine selector (PaddleOCR)

## CI/CD

GitHub Actions builds and pushes the Docker image to Azure Container Registry on merge to main, then deploys to App Service. The AKS deployment uses the same image from ACR, deployed separately via Helm.

## Tools Used

- **Tesseract** — primary OCR engine
- **PaddleOCR** (PP-OCRv4) — experimental OCR engine (debug mode)
- **Quart** — async Python web framework
- **OpenCV** — image preprocessing
- **Bootstrap 5** — frontend UI
- **Docker** — containerization
- **Helm** — Kubernetes deployment
- **GitHub Actions** — CI/CD

## Assumptions & Trade-offs

- **No COLA integration** — This is a standalone prototype per Marcus's guidance. The API contract (`POST /api/verify`) is designed to be straightforward to integrate with an upstream system in the future.
- **No authentication** — Out of scope for the prototype. A production deployment would require integration with the agency's identity provider.
- **No persistent storage** — Labels are processed in-memory and not retained. This sidesteps document retention and PII concerns for the prototype phase.
- **OCR quality depends on image quality** — Degraded images (extreme angles, heavy glare, very low resolution) will produce lower confidence results. The system surfaces this honestly via confidence scores rather than guessing.
- **Government warning detection is presence-based** — The system checks that the required warning segments are present in the OCR text. It does not currently verify exact formatting (bold, all-caps for "GOVERNMENT WARNING:") since OCR doesn't preserve typographic attributes. Noted as a V2 enhancement.
- **Benchmark dataset is small** — The included sample labels demonstrate the pipeline end-to-end. A production system would require evaluation against a much larger and more diverse dataset.

## What's Next

See `V2_ROADMAP.md` for planned improvements including expanded dataset evaluation, layout-aware field detection, and formatting verification.
