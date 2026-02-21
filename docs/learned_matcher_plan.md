# Learned Pair Scorer — Design & Implementation Plan

## 1. The Problem

After YOLO detects chemical structures (class 0) and compound labels (class 1) on a
document page, there is no inherent connection between the two sets of detections.
The goal of the **matcher** is to produce a bijection S → L: assign each detected
structure to its correct label.

The current `HungarianMatcher` solves this by building a cost matrix where
`cost[i,j] = euclidean_distance(centroid(sᵢ), centroid(lⱼ))` and solving the optimal
1-to-1 assignment via `scipy.optimize.linear_sum_assignment`.

This is a sound baseline. It is not wrong. But it has a hard ceiling.

---

## 2. Why Euclidean Hungarian Fails

### 2.1 The Symmetric Grid Problem

Consider a 3×3 grid of compound panels, all uniformly spaced. Structure s₁ is equally
far from label l₁ (its true partner, directly below) and label l₂ (the next column's
label, also directly below but one column over). Euclidean distance cannot distinguish
the correct pair from the wrong one. The Hungarian algorithm resolves the tie
arbitrarily.

```
[s1]  [s2]  [s3]
[l1]  [l2]  [l3]    ← equidistant: dist(s1,l1) ≈ dist(s1,l2)
[s4]  [s5]  [s6]
[l4]  [l5]  [l6]
```

### 2.2 Directional Blindness

Labels appear below a structure 50% of the time, above 20%, left 15%, right 15% (from
the generator distribution). A label that is slightly farther away but directly below a
structure is almost certainly its partner. A label that is equidistant but to the left
is much less likely to be the partner. Euclidean distance ignores direction entirely.

### 2.3 Scale Blindness

A large structure (600px wide) and a small structure (200px wide) at the same centroid
distance from the same label have very different semantic proximity. The label that is
"50px below a 600px structure" and "50px below a 200px structure" carry very different
confidences. Normalizing distance by structure size is the natural fix, but Euclidean
distance does not do this.

### 2.4 Page Position Blindness

Near the bottom edge of a page, no label can appear below a structure — it would fall
off the page. The generator's label placement logic respects this. A Euclidean matcher
does not encode this constraint.

### 2.5 Distractor Confusion

YOLO may detect non-label text regions (captions, equations, property values, reaction
conditions) as class-1 labels. These distractors are closer to some structure than the
true label is. Euclidean matching will assign the structure to the distractor. A matcher
with visual awareness of the label crop (is this actually a short compound identifier or
a long caption fragment?) can reject distractors.

---

## 3. Why YOLO Keypoint Prediction Does Not Solve This

An appealing alternative is to add a keypoint prediction head to the structure detector:
"for each detected structure, predict the (x, y) offset to its associated label center."
This would bypass the assignment problem entirely.

This approach has a fundamental epistemological flaw: **there is no information in or
around a molecule's bounding box that tells the model where the layout algorithm placed
its label.**

The label `CHEMBL2845791` is an arbitrary database identifier. It could be above, below,
left, or right of any molecule. The molecule's visual appearance carries zero information
about this. The keypoint head would converge to one of three degenerate behaviors:

1. **Learn the prior distribution.** It outputs "label is roughly 80px below" for every
   structure, because 50% of labels are below in training data. This is a hardcoded
   heuristic dressed up as ML. It produces the same answer regardless of context.

2. **Detect nearby text via receptive field.** YOLO11's large effective receptive field
   means the backbone features at a structure's center may "see" nearby text blobs. The
   model learns to fire the keypoint at the nearest text blob. This is implicit label
   detection, not learned association. It fails when multiple text blobs are nearby
   (dense grids) and when the tile boundary splits structure from label.

3. **Learn layout type.** In grid pages, all labels are below all structures. The model
   might learn "this is a grid page, fire all keypoints downward." This requires global
   context that a 1536px tile does not always provide.

None of these constitute a meaningful structure→label association. The approach is
borrowed from human pose estimation where the joint positions are causally determined
by body geometry — a condition that does not hold here. **The correct inductive bias
for this problem requires seeing both the structure and the label simultaneously.**

---

## 4. Tier 1.5: Learned Pair Scoring

### 4.1 Core Idea

Train a binary classifier that takes a candidate (structure, label) pair and outputs
a probability: "how likely is this pair a true association?"

At inference, compute this score for every possible (structure, label) pair on the
page. Use the scores as the cost matrix for Hungarian matching:

```
cost[i,j] = 1.0 - scorer(sᵢ, lⱼ)
```

The optimal assignment under learned cost replaces the Euclidean cost. Hungarian is
preserved — it still guarantees a globally optimal 1-to-1 assignment. Only the cost
metric changes.

### 4.2 Why This Has the Right Inductive Bias

The scorer receives features from both objects simultaneously. It learns:
- The joint distribution of (structure position, label position) from thousands of pages
- That dx/struct_width is a better proximity measure than raw dx
- That a label directly below at distance 60px is more likely correct than one to the
  side at distance 40px
- That near the bottom of the page, the label is expected above, not below
- That a narrow tall crop is more likely a label identifier than a wide low crop
  (which might be a caption fragment)

None of this requires the model to "know" which molecule owns which label from their
appearances. It learns the statistical regularity of how layout algorithms place labels
relative to structures. This is the correct thing to learn.

### 4.3 Feature Vector

#### Geometric Features (Primary Signal, Always Used)

For a candidate pair `(sᵢ, lⱼ)` on a page of size `(W, H)`:

| Feature | Formula | Rationale |
|---|---|---|
| `dx_norm` | `(lⱼ_cx - sᵢ_cx) / sᵢ_width` | Lateral offset normalized to structure size |
| `dy_norm` | `(lⱼ_cy - sᵢ_cy) / sᵢ_height` | Vertical offset normalized to structure size |
| `dist_norm` | `√(dx_norm² + dy_norm²)` | Scale-invariant distance |
| `angle_sin` | `sin(atan2(dy_norm, dx_norm))` | Direction component (avoid discontinuity) |
| `angle_cos` | `cos(atan2(dy_norm, dx_norm))` | Direction component |
| `size_ratio` | `(lⱼ_area) / (sᵢ_area)` | Label is typically 3–8% of structure area |
| `label_aspect` | `lⱼ_width / lⱼ_height` | Compound IDs are wide/flat; captions are very wide |
| `struct_page_x` | `sᵢ_cx / W` | Horizontal page position (edge constraint awareness) |
| `struct_page_y` | `sᵢ_cy / H` | Vertical page position (bottom edge → label above) |
| `label_page_x` | `lⱼ_cx / W` | |
| `label_page_y` | `lⱼ_cy / H` | |
| `label_conf` | `lⱼ.conf` | YOLO confidence for label detection |
| `struct_conf` | `sᵢ.conf` | YOLO confidence for structure detection |

**Total: 13 geometric features.**

Why sin/cos for angle: `atan2` has a discontinuity at ±π. Encoding as (sin, cos) gives
a smooth, periodic representation that the MLP can learn from without needing the
discontinuity.

#### Visual Features (Secondary Signal, Improves Distractor Rejection)

Crop each region from the page image. Resize to a fixed size. Encode with a small CNN.

- **Structure crop**: resize to `128×128` → small CNN → 64-d embedding
- **Label crop**: resize to `96×32` (wide aspect for text) → small CNN → 32-d embedding

The structure visual embedding primarily helps distinguish real structures from
false-positive detections. The label visual embedding helps distinguish true compound
identifiers (short, alphanumeric, specific fonts) from distractor text (long captions,
equations, property values).

The model will also learn whether the spatial context of the label crop looks like
it belongs near this structure (e.g., consistent with the overall page density).

### 4.4 Model Architecture

```
struct_crop [128×128×1]  →  SmallCNN  →  64-d
label_crop  [ 96×32×1]  →  SmallCNN  →  32-d
geom_feats  [13-d]       →  Linear(13→32)  →  32-d

concat [64 + 32 + 32] = 128-d
    ↓
Linear(128 → 64) + ReLU + Dropout(0.2)
    ↓
Linear(64 → 32) + ReLU
    ↓
Linear(32 → 1) + Sigmoid
    ↓
association_score ∈ [0, 1]
```

**SmallCNN** (shared architecture, separate weights for struct/label):
```
Conv2d(1→32, 3×3) + BN + ReLU
MaxPool(2×2)
Conv2d(32→64, 3×3) + BN + ReLU
MaxPool(2×2)
Conv2d(64→128, 3×3) + BN + ReLU
AdaptiveAvgPool → 128-d
Linear(128 → output_dim)
```

For a geometry-only baseline (Tier 1 proper), drop the CNNs and input just the 13
geometric features into the MLP directly. This is faster and simpler, and may be
sufficient. The visual branch is additive.

### 4.5 Training Data Construction

#### Source
30,000 pages × ~5 structures/page ≈ **150,000 positive pairs** already exist in the
ground truth JSON files. No new data generation is needed.

#### Positive Pairs
Every `(struct_bbox[i], label_bbox[i])` entry in the ground truth JSON where
`label_bbox` is not null.

#### Negative Pairs
All other cross-pairings of structures and labels on the same page. For a page with
5 structures and 5 labels, there are 5 positive pairs and 20 negative pairs.

**Hard negative sampling**: Random sampling of all negatives would be dominated by
trivially easy cases (structures on opposite corners of the page). Sample negatives
weighted by inverse distance — prefer negatives where the wrong label is spatially
close to the structure. These are the cases that fool the Euclidean matcher and where
learning adds most value.

A practical rule: for each positive pair `(sᵢ, lᵢ)`, generate hard negatives by
pairing `sᵢ` with the 2–3 spatially nearest wrong labels on the page.

#### Class Imbalance
Positive:Negative ratio is approximately 1:4 (with hard negative sampling). Use
`pos_weight=4.0` in `BCEWithLogitsLoss` to compensate.

#### Using GT Bboxes vs. Predicted Bboxes
Training on perfect ground truth bboxes may cause a distribution shift at inference
where YOLO produces slightly imprecise bboxes. Apply coordinate jitter during training:

```
struct_bbox + Uniform(-0.02 * w, +0.02 * w) per coordinate
label_bbox  + Uniform(-0.02 * w, +0.02 * w) per coordinate
```

This simulates YOLO localization noise and makes the scorer robust to it.

#### Train / Val Split
Use pages from the existing train/val split:
- Training: 30,000 pages → ~150,000 positive pairs
- Validation: 5,000 pages → ~25,000 positive pairs

Do not mix pages across splits (to prevent data leakage).

### 4.6 Training Procedure

```
Optimizer:   AdamW, lr=1e-3, weight_decay=1e-4
Scheduler:   CosineAnnealingLR, T_max=30
Batch size:  2048 pairs (fast with A6000)
Epochs:      30
Loss:        BCEWithLogitsLoss(pos_weight=4.0)
Metrics:     Binary accuracy, Precision, Recall, F1, AUC-ROC
             Pair-level accuracy: fraction of pages where ALL pairs correct
```

Hard negative mining after epoch 3: re-score training negatives, promote the 10%
with highest score to "hard negatives" — oversample these in subsequent epochs.

### 4.7 Evaluation Metrics

Standard binary classification metrics are insufficient here. What matters is whether
the final assignment (after Hungarian) is correct, not whether individual pair scores
are calibrated.

**Primary metric: Page-level Pair Accuracy**
For a page with n structures and n labels, the assignment is correct if and only if
all n pairs are correctly matched. Report the fraction of pages where all pairs are
correct on the validation set.

Compare directly against `HungarianMatcher` on the same validation images.

**Secondary metrics**:
- Per-pair precision/recall on the scored pairs (before Hungarian)
- AUC-ROC of the scorer
- Error analysis by page type (grid, free-form, SAR, MMP, Excel, etc.)

---

## 5. Inference: LearnedMatcher

### 5.1 Algorithm

```python
# 1. Separate detections by class
structures = [d for d in detections if d.class_id == 0]
labels     = [d for d in detections if d.class_id == 1]

# 2. Score all n×m pairs
score_matrix = np.zeros((len(structures), len(labels)))
for i, s in enumerate(structures):
    for j, l in enumerate(labels):
        score_matrix[i, j] = scorer(s, l, image, page_size)

# 3. Hungarian on inverted scores
cost_matrix = 1.0 - score_matrix
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# 4. Threshold: reject pairs below confidence
pairs = []
for r, c in zip(row_ind, col_ind):
    if score_matrix[r, c] >= min_score:
        pairs.append(CompoundPair(
            structure=structures[r],
            label=labels[c],
            match_distance=1.0 - score_matrix[r, c],  # lower=better, consistent with HungarianMatcher
        ))
```

The `match_distance` field is repurposed as `1 - score` so that the semantics remain
consistent: lower distance = higher confidence match.

### 5.2 Null Label Handling

If `score_matrix[r, best_c] < min_score` for structure r, the structure is considered
unlabeled. This replaces the `max_distance` threshold in `HungarianMatcher`.

### 5.3 Complexity

For a typical page with n≤10 structures and n≤10 labels, the pair scoring is 100
forward passes through a tiny model. This is negligible on GPU. Even on CPU, a
geometry-only scorer (no CNN) is essentially free.

For pages with many detections, optionally pre-filter the candidate pairs by a loose
Euclidean threshold (e.g., 3× the median structure width) before scoring.

---

## 6. Technical Changes Required

### 6.1 New Package: `structflo/cser/lps/`

All learned pair scorer code — training and inference — lives here, inside the
`structflo/cser` package where it belongs.

```
structflo/cser/lps/
├── __init__.py      # re-exports LearnedMatcher for clean consumer imports
├── features.py      # Feature extraction: (Detection, Detection, image, page_wh) → tensor
│                    # Shared kernel: called by dataset.py (training) and matcher.py (inference)
├── scorer.py        # nn.Module definitions: GeomScorer, VisualScorer (no I/O, no training logic)
├── dataset.py       # PyTorch Dataset: load GT JSON + images → (feature_tensor, label)
├── train.py         # Training loop + main() → entry point sl-train-lps
├── evaluate.py      # Page-level pair accuracy: LearnedMatcher vs HungarianMatcher on val set
└── matcher.py       # LearnedMatcher(BaseMatcher) — the inference adapter
```

**Dependency directions (no circular imports):**
- `lps/features.py` — pure functions, no pipeline dependency
- `lps/scorer.py` — `nn.Module` only, no pipeline dependency
- `lps/dataset.py` — imports `features.py`, reads GT JSON files
- `lps/matcher.py` — imports `BaseMatcher`, `Detection`, `CompoundPair` from `pipeline/`; imports `features.py` and `scorer.py` from within `lps/`
- `lps/train.py` — imports `dataset.py`, `scorer.py`; no pipeline dependency
- `lps/__init__.py` — `from lps.matcher import LearnedMatcher`
- `pipeline/` — does **not** import from `lps/` (clean one-way dependency)

**Consumer usage:**
```python
from structflo.cser.lps import LearnedMatcher
pipeline = ChemPipeline(matcher=LearnedMatcher("runs/lps/scorer_best.pt"))
```

### 6.2 `LearnedMatcher` in `lps/matcher.py`

```python
class LearnedMatcher(BaseMatcher):
    def __init__(
        self,
        weights: Path | str,          # path to trained scorer .pt checkpoint
        min_score: float = 0.5,       # pairs below this score → unmatched
        device: str = "cuda",
        geometry_only: bool = False,  # skip visual CNN branch
    ): ...

    def match(self, detections: list[Detection], image: np.ndarray | None = None) -> list[CompoundPair]:
        ...
```

### 6.3 Interface Extension: `BaseMatcher`

The current signature:
```python
def match(self, detections: list[Detection]) -> list[CompoundPair]:
```

Should become:
```python
def match(self, detections: list[Detection], image: np.ndarray | None = None) -> list[CompoundPair]:
```

`HungarianMatcher` ignores the `image` argument. `LearnedMatcher` uses it for visual
crops. This is backwards compatible.

### 6.4 `CompoundPair` model — no change required

`match_distance` remains as-is. `LearnedMatcher` sets it to `1.0 - score`, preserving
the "lower = better" convention. Downstream consumers see no difference.

Optionally add `match_score: float | None = None` if callers need the raw score
distinct from the normalized distance. This is additive and backwards compatible.

### 6.5 `ChemPipeline` — no change required

The pipeline already accepts a `matcher: BaseMatcher` constructor argument. Swapping
matchers is already designed in:

```python
# Existing (default)
pipeline = ChemPipeline()

# New
from structflo.cser.pipeline.learned_matcher import LearnedMatcher
pipeline = ChemPipeline(
    matcher=LearnedMatcher(weights="runs/matcher/scorer.pt", min_score=0.5)
)
```

### 6.6 CLI Extension: `sf-extract`

Add optional flags:
```
--matcher {hungarian,learned}    default: hungarian
--matcher-weights PATH           path to learned scorer checkpoint
--matcher-min-score FLOAT        default: 0.5, rejection threshold for LearnedMatcher
--matcher-geometry-only          use geometric features only (no visual CNN)
```

### 6.7 New Entry Point: `sl-train-matcher`

```
sl-train-matcher  →  struct_labels.matching.train:main
```

Arguments:
```
--data-dir PATH          root of generated data (train/ val/ subdirs with ground_truth/)
--output-dir PATH        where to save checkpoints (default: runs/matcher/)
--epochs INT             default: 30
--batch-size INT         default: 2048
--geometry-only          train geometry-only model (no visual branch)
--device STR             default: cuda
```

### 6.8 Weights Registry: `weights.py`

The existing `weights.py` registry and `resolve_weights()` API should be extended with
a new model entry for the LPS scorer. No new mechanism is needed — this is a straight
addition.

**Add to `REGISTRY`:**
```python
REGISTRY: dict[str, dict[str, dict]] = {
    "cser-detector": { ... },          # existing

    "cser-lps": {                      # new
        "v1.0": {
            "repo_id":  "sidxz/structflo-cser-lps",
            "filename": "scorer_best.pt",
            "revision": "weights-v1.0",
            "sha256":   "...",         # fill after first publish
            "requires": ">=0.1.0,<1.0.0",
        },
    },
}
```

**Add to `LATEST`:**
```python
LATEST: dict[str, str | None] = {
    "cser-detector": "v0.1",
    "cser-lps":      None,    # set to "v1.0" after first publish
}
```

`LearnedMatcher` calls `resolve_weights("cser-lps", version=weights)` using the same
logic as the detector. The `weights` parameter follows the same three-mode convention:

- `None` → LATEST (auto-download)
- `"v1.0"` → specific registered version (auto-download)
- `"/path/to/scorer.pt"` → local file, no download

**Publishing workflow (after training):**
1. Train scorer → `runs/lps/scorer_best.pt`
2. Create HF Hub repo `sidxz/structflo-cser-lps`
3. Push `scorer_best.pt`, create git tag `weights-v1.0`
4. Fill in `sha256` in the registry entry
5. Set `LATEST["cser-lps"] = "v1.0"`

### 6.9 `pyproject.toml` additions

```toml
[project.scripts]
sl-train-lps = "structflo.cser.lps.train:main"

[project.optional-dependencies]
lps = ["torch", "torchvision"]
```

`torch` and `torchvision` are likely already installed; listing under `lps` extra
makes the dependency explicit and allows `pip install structflo-cser[lps]` for
environments that only need inference (the scorer weights are loaded from HF Hub,
but training is optional).

---

## 7. Data Flow Summary

### Training Flow

```
data/generated/train/
├── images/        train_000000.jpg  ...
└── ground_truth/  train_000000.json ...
        ↓
dataset.py: for each page
    load JSON → positive pairs (struct_bbox, label_bbox)
    generate hard negatives (wrong cross-pairs, proximity-weighted)
    apply bbox jitter (simulate YOLO noise)
    extract geometric features (13-d)
    optionally crop image for visual features
        ↓
scorer.py: GeomScorer or VisualScorer
        ↓
train.py: BCEWithLogitsLoss + AdamW + cosine LR
    epoch 1–3: random negatives
    epoch 4+:  hard negative mining
        ↓
runs/matcher/
├── scorer_best.pt
├── scorer_final.pt
└── training_log.csv
```

### Inference Flow

```
page image
    ↓
ChemPipeline.detect() → list[Detection] (structures + labels)
    ↓
LearnedMatcher.match(detections, image)
    → for each (sᵢ, lⱼ) candidate pair:
        extract geometric features
        optionally crop image regions for visual features
        scorer forward pass → score[i,j]
    → cost_matrix = 1.0 - score_matrix
    → linear_sum_assignment(cost_matrix)
    → filter by min_score
    → list[CompoundPair]
    ↓
ChemPipeline.enrich() → SMILES + OCR
    ↓
Output
```

---

## 8. Implementation Order

1. **`struct_labels/matching/features.py`**
   Implement geometric feature extraction. Unit test against known GT pairs.

2. **`struct_labels/matching/dataset.py`**
   Load GT JSONs, generate positive/negative pairs, yield feature tensors.
   Verify class balance and hard negative distribution.

3. **`struct_labels/matching/scorer.py`**
   Implement `GeomScorer` (geometry only) and `VisualScorer` (+ CNN branches).
   Start with `GeomScorer` — it is faster to iterate on.

4. **`struct_labels/matching/train.py`**
   Training loop, hard negative mining after epoch 3, checkpoint saving.
   Validate with page-level pair accuracy on val set.

5. **`struct_labels/matching/evaluate.py`**
   Compare `HungarianMatcher` vs `LearnedMatcher` on validation pages.
   Report per-page-type breakdown.

6. **`structflo/cser/pipeline/learned_matcher.py`**
   Implement `LearnedMatcher(BaseMatcher)`. Load checkpoint, extract features,
   score matrix, Hungarian assignment.

7. **Extend `BaseMatcher` and `sf-extract` CLI** with optional `image` argument
   and `--matcher` flags.

8. **Smoke test end-to-end**: run `sf-extract` with `--matcher learned` on 10 real
   pages, compare output against `--matcher hungarian`.

---

## 9. Open Questions

- **Geometry-only vs Visual**: Train both variants. The geometry-only model is the
  simpler bet and is likely sufficient for 90%+ of pages. The visual branch adds
  training complexity. Evaluate both before committing to the visual branch.

- **GT bboxes vs predicted bboxes for training**: Training on GT bboxes is simpler.
  If the geometry-only scorer is sensitive to localization noise, re-train on YOLO
  predictions from the training set (run inference, save predicted bboxes, train
  scorer on those). Jitter augmentation is a cheap proxy for this.

- **Min score threshold calibration**: Tune `min_score` on the validation set to
  balance false positives (wrong pair accepted) vs false negatives (correct pair
  rejected because score is slightly below threshold). The null-label rate (~10%)
  sets a natural prior.

- **Page type distribution shift**: The val set includes all 8 page types. Report
  accuracy broken down by page type. Grid and Excel pages (most ambiguous geometry)
  will be the hardest; those are the primary motivation for this work.
