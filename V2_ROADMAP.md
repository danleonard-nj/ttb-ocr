# V2 Roadmap

## Current State (V1)

Rule-based OCR pipeline with regex extraction and token matching. Works well for clean, standardized labels but brittle against varied layouts, handwritten elements, and non-standard label designs.

**V1 limitations:**

- Fixed regex patterns break on unexpected formats
- Token overlap matching can't handle synonyms or abbreviations it hasn't seen
- Government warning detection relies on exact phrase anchors
- No learning from corrections — every edge case requires a manual rule
- No label-level classification (wine vs. beer vs. spirits all use the same logic)

---

## V2: Learned Matching + Human-in-the-Loop

### 1. Label Classifier

Pre-classify the label type before running extraction. Different label categories have different layouts, font sizes, and regulatory patterns.

**Approach:**

- Lightweight CNN or vision model (e.g. MobileNet fine-tuned) to classify: wine / beer / spirits / RTD / other
- Classification drives downstream extraction strategy (which regions to crop, which patterns to prioritize)
- Training data: existing `master.csv` has `Class/Type` ground truth for supervised training

**Why it matters:** A wine label and a craft beer can have completely different layouts. Knowing the type up front lets the extractor focus on the right regions.

### 2. TF-IDF / Learned Text Labeling

Replace rigid regex field extraction with a trained text labeling model.

**Approach:**

- Extract all OCR text spans with bounding boxes (already have this)
- Feature engineering: TF-IDF over text content, positional features (x/y normalized coordinates, relative size), font size proxy (box height), spatial context (neighboring text)
- Train a classifier (random forest, gradient boosting, or small NN) to label each text span: `brand_name` / `alcohol_content` / `net_contents` / `gov_warning` / `other`
- Fall back to current regex logic for low-confidence predictions

**Training data source:** V1 results with human corrections become the training set. Each verified label produces labeled text spans.

### 3. LLM-Assisted Parsing

Use an LLM for the hard cases that rule-based and ML approaches miss.

**Approach:**

- When confidence is below threshold on any field, send the full OCR text + image context to an LLM
- Structured output prompt: "Given this OCR text from an alcohol beverage label, extract: brand_name, alcohol_content, net_contents, government_warning_present"
- LLM handles ambiguity, abbreviations, and context that regex can't (e.g. "ALC 11.5% BY VOL" vs "ALCOHOL 11.5 PERCENT")
- Use as a **fallback tier**, not primary — keeps costs and latency controlled

**Integration:** OcrService gets a new fallback tier after engine fallbacks. Only fires when fields are still missing/low-confidence after ML extraction.

### 4. Richer Historical Training Data

Scale the training set beyond the current small dataset.

**Approach:**

- Expand scraping pipeline to pull more TTB records (currently limited by `LIMIT` config)
- Include distorted/augmented variants (pipeline already supports blur, rotation, noise, perspective warp)
- Incorporate historical label approvals (TTB public records go back years)
- Cross-reference with open datasets (OpenFoodFacts, etc.) for additional ground truth

**Target:** 1,000+ labeled images across diverse label types, sufficient for training the classifier and text labeler.

### 5. Human-in-the-Loop

Close the feedback loop: human corrections feed back into model training.

**Approach:**

- **Review queue UI:** Surface low-confidence results for human review
- **Correction interface:** Show annotated image with extracted fields, let reviewer correct any field in-place
- **Active learning:** Prioritize review of predictions the model is least certain about
- **Feedback pipeline:** Corrected labels get written back to training set, trigger periodic model retraining
- **Audit trail:** Log all corrections with reviewer ID, timestamp, original vs. corrected values

**UI integration:** Add a `/review` page alongside the existing `/verify` page. Reviewers see a queue of labels sorted by confidence (lowest first). Each correction is a training example.

---

## Implementation Priority

| Phase  | Work                                              | Impact                                        |
| ------ | ------------------------------------------------- | --------------------------------------------- |
| **2a** | Human-in-the-loop review UI + correction pipeline | Unlocks training data generation              |
| **2b** | TF-IDF text span labeler                          | Replaces brittle regex, adapts to new layouts |
| **2c** | Label classifier                                  | Optimizes extraction per label type           |
| **2d** | LLM fallback tier                                 | Catches long-tail edge cases                  |
| **2e** | Dataset expansion (1k+ images)                    | Improves all learned components               |

Human-in-the-loop comes first because it produces the training data everything else depends on.

---

## V2 Pipeline Architecture

V1's `OcrService.evaluate_image()` is a single method that hardcodes the full flow. V2 refactors this into a staged pipeline where each step receives and enriches a shared context object.

### Pipeline Context

A `PipelineContext` dataclass flows through every stage:

```python
@dataclass
class PipelineContext:
    image_path: str
    image: np.ndarray | None = None
    label_type: str | None = None          # V2c: wine / beer / spirits / other
    ocr_attempts: list[dict] = field(default_factory=list)
    merged_text: str = ""
    text_spans: list[TextSpan] = field(default_factory=list)  # bounding box + text
    fields: dict[str, FieldResult] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

### Pipeline Stages

```
Pipeline(steps=[
    LoadImage,          # Read image, validate, set context.image
    ClassifyLabel,      # V2c — CNN classifies wine/beer/spirits/other
    OcrExtract,         # Run engine(s), populate ocr_attempts + merged_text
    FieldExtract,       # V2b — TF-IDF text span labeler (falls back to regex)
    MatchFields,        # Compare extracted values against expected
    LlmFallback,       # V2d — LLM fills gaps when confidence < threshold
    ScoreResult,        # Aggregate per-field confidence into overall score
])
```

### Stage Interface

Each stage implements a simple protocol:

```python
class PipelineStage(Protocol):
    def run(self, ctx: PipelineContext) -> PipelineContext: ...
    def should_run(self, ctx: PipelineContext) -> bool: ...
```

`should_run` enables conditional execution — e.g., `LlmFallback` only fires when fields are missing or below a confidence threshold. Stages are composable: adding a new stage means appending to the list, not modifying existing code.

### Migration Path from V1

| V1 Component                      | V2 Stage              | Change Required                                        |
| --------------------------------- | --------------------- | ------------------------------------------------------ |
| `OcrService.evaluate_image()`     | `OcrExtract`          | Extract OCR loop into standalone stage                 |
| `LabelMatcher.match_expected_*()` | `MatchFields`         | Wrap existing matcher as one strategy behind interface |
| `LabelMatcher.evaluate_text()`    | `FieldExtract`        | Regex becomes fallback; TF-IDF becomes primary         |
| `_should_fallback()` + retry loop | `Pipeline.run()` flow | Replaced by stage-level `should_run` checks            |
| _(new)_                           | `ClassifyLabel`       | CNN classifier added as pre-extraction step            |
| _(new)_                           | `LlmFallback`         | LLM tier added as post-extraction fallback             |

### Why This Structure

- **Testable:** Each stage can be unit-tested in isolation with a mock `PipelineContext`.
- **Composable:** New capabilities (e.g., barcode reader, layout detector) are added as stages without touching existing code.
- **Observable:** Each stage can log its own timing and confidence delta, making it easy to identify bottlenecks.
- **Incrementally adoptable:** V1 logic wraps into stages without rewriting — the migration is structural, not functional.

---

## Non-Goals for V2

- **Full government warning OCR with bounding box detection** — Detecting the complete warning text block as a single region would require either a trained object detection model (YOLO/Faster R-CNN fine-tuned on label regions) or a connected-component analysis pipeline that groups nearby text spans. Both approaches significantly increase complexity for marginal gain over the current multi-anchor pattern matching approach. Revisit in V3 if the text labeler naturally clusters warning spans. See analysis below.

---

## Appendix: Government Warning Bounding Box Analysis

### Current Approach

V1 detects government warning presence by matching text anchors ("government warning", "surgeon general", "pregnancy", "impairment") across all OCR text. Individual word-level bounding boxes that contain these keywords are associated back to the `gov_warning` field for display.

### What "Whole Label Detection" Would Require

To detect the government warning as a **single bounding box** encompassing the entire warning block:

**Option A: Spatial Clustering**

- Group OCR bounding boxes by spatial proximity (DBSCAN or similar)
- Identify the cluster containing government warning anchors
- Compute the convex hull / bounding rect of that cluster
- **Problem:** Warnings are often interleaved with other text, printed in tiny font, or split across columns. Clustering frequently grabs neighboring non-warning text.

**Option B: Object Detection Model**

- Fine-tune YOLO or Faster R-CNN to detect "government warning region" as a class
- Requires 500+ annotated training images with bounding box labels on the warning block
- **Problem:** Annotation cost is high, and the warning block varies enormously across labels (size, position, font, orientation).

**Option C: Connected Component + Heuristic**

- After OCR, find the first anchor ("GOVERNMENT WARNING"), then greedily expand the bounding box downward/rightward to include subsequent lines until text stops matching warning phrases
- **Problem:** Relies on OCR line ordering being correct and spatially consistent. Falls apart on curved, rotated, or multi-column labels.

### Recommendation

Stick with the current per-keyword box association for V1/V2. The visual overlay already highlights where warning text was found. A dedicated warning region detector is a V3 feature that benefits from the labeled training data produced by the human-in-the-loop pipeline.
