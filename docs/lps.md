# Learned Pair Scorer (LPS)

Reference documentation for `structflo/cser/lps/` — the learned structure-label
association module that replaces or augments Euclidean Hungarian matching.

---

## 1. Why

After YOLO detects chemical structures (class 0) and compound labels (class 1), a
**matcher** must assign each structure to its correct label. The default
`HungarianMatcher` uses centroid Euclidean distance as the cost. This fails in four
identifiable ways:

| Failure mode | Example |
|---|---|
| **Symmetric grids** | 3×3 grid: `dist(s1,l1) ≈ dist(s1,l2)` — tie broken arbitrarily |
| **Directional blindness** | A label 60px directly below is more likely correct than one 40px to the side |
| **Scale blindness** | 50px gap beside a 600px structure ≠ 50px gap beside a 200px structure |
| **Page-edge constraint** | Near the bottom of a page, labels must appear above, not below |

A learned scorer can internalize all of these from training data because it sees both
objects simultaneously and can learn the statistical regularity of how layout algorithms
place labels relative to structures.

The design **preserves Hungarian matching** — only the cost metric changes:

```
cost[i,j] = 1.0 - scorer(sᵢ, lⱼ)
```

---

## 2. What Was Rejected and Why

**YOLO keypoint prediction** (predict dx,dy offset from structure to label) is
fundamentally flawed: there is no information in a molecule's bounding box that tells
the model where the layout algorithm placed its label. A compound ID like `CHEMBL2845791`
is an arbitrary string that can appear anywhere around the structure. The keypoint head
would degenerate to learning the prior ("labels are usually below") or detecting nearby
text via receptive field — neither constitutes true association. See
`learned_matcher_plan.md` §3 for the full epistemological argument.

**Instance segmentation (Mask R-CNN)** still requires the RPN to propose the
union bounding box (structure + whitespace + label). This reintroduces the same
whitespace-center problem as union_bbox detection, at the proposal stage.

---

## 3. Feature Engineering

### 3.1 Geometric features — 14-d vector

Computed by `structflo/cser/lps/features.py:geom_features()`.

| # | Feature | Formula | Why |
|---|---|---|---|
| 0 | `dx_norm` | `(l_cx − s_cx) / s_w` | Lateral offset, structure-size-normalised |
| 1 | `dy_norm` | `(l_cy − s_cy) / s_h` | Vertical offset, structure-size-normalised |
| 2 | `dist_norm` | `√(dx_norm² + dy_norm²)` | Scale-invariant distance |
| 3 | `angle_sin` | `sin(atan2(dy_norm, dx_norm))` | Smooth angular encoding (no ±π jump) |
| 4 | `angle_cos` | `cos(atan2(dy_norm, dx_norm))` | — |
| 5 | `size_ratio` | `l_area / s_area` | Label is typically 3–8% of structure area |
| 6 | `label_aspect` | `l_w / l_h` | Compound IDs are wide; captions are very wide |
| 7 | `struct_aspect` | `s_w / s_h` | Structure shape |
| 8 | `struct_page_x` | `s_cx / W` | Horizontal position on page |
| 9 | `struct_page_y` | `s_cy / H` | Vertical position; near bottom → label above |
| 10 | `label_page_x` | `l_cx / W` | — |
| 11 | `label_page_y` | `l_cy / H` | — |
| 12 | `struct_conf` | YOLO confidence | Detection quality signal |
| 13 | `label_conf` | YOLO confidence | Detection quality signal |

### 3.2 Visual features (optional)

When `--visual` is used, two grayscale image crops are extracted and encoded by small
CNNs:

- **Structure crop**: `128×128` → SmallCNN → 64-d embedding
- **Label crop**: `32×96` (tall-to-wide for text) → SmallCNN → 32-d embedding

The label visual branch mainly helps distinguish true compound IDs (short, alphanumeric,
specific fonts) from distractor text (long captions, equations, property values).

---

## 4. Models

All in `structflo/cser/lps/scorer.py`. Both output raw logits (apply sigmoid for
probabilities).

### GeomScorer (default, recommended)

```
Linear(14 → 128) + LayerNorm + ReLU
Linear(128 → 64) + LayerNorm + ReLU + Dropout(0.1)
Linear(64 → 32) + ReLU
Linear(32 → 1)
```

~12K parameters. Trains in minutes. Sufficient for the vast majority of pages.

### VisualScorer

```
struct_crop [1×128×128]  →  SmallCNN  →  64-d
label_crop  [1×32×96]   →  SmallCNN  →  32-d
geom_feats  [14-d]       →  Linear(14→32)  →  32-d

concat [128-d]
    → Linear(128→64) + BN + ReLU + Dropout(0.2)
    → Linear(64→32) + ReLU
    → Linear(32→1)
```

~224K parameters. Adds distractor-rejection capability at the cost of image I/O during
training and inference.

---

## 5. Training Data

### Source

`data/generated/train/` and `data/generated/val/` — the same synthetic pages used to
train the YOLO detector.

```
data/generated/
├── train/
│   ├── images/           30,000 × A4@300DPI JPEGs
│   └── ground_truth/     30,000 × JSON  (struct_bbox, label_bbox, smiles, …)
└── val/
    ├── images/            5,000 × A4@300DPI JPEGs
    └── ground_truth/      5,000 × JSON
```

### Pair construction

`LPSDataset._build()` processes each GT JSON:

**Positive pairs** — every `(struct_bbox[i], label_bbox[i])` where `label_bbox` is not
null. ~5 positives per page → ~150K train, ~25K val.

**Hard negatives** — for each positive `(sᵢ, lᵢ)`, pair `sᵢ` with the `neg_per_pos`
(default 3) spatially nearest *wrong* labels on the same page. These are the pairs that
fool Euclidean matching and where the scorer adds most value.

Ratio: 1 positive : 3 negatives → `pos_weight ≈ 3.0` passed to `BCEWithLogitsLoss`.

### Bbox jitter (train only)

Simulates YOLO localisation noise so the scorer is robust to imprecise predicted bboxes
at inference:

```python
coord += Uniform(-jitter * side_length, +jitter * side_length)   # jitter=0.02
```

---

## 6. Dataset Implementation — Performance Notes

`structflo/cser/lps/dataset.py` was designed for fast multiprocessing.

### Why compact numpy arrays

The dataset stores all metadata as flat numpy arrays (`int32`, `float32`, `int8`) rather
than Python objects. ~700K samples → ~40MB pickle vs ~350MB for equivalent Python
objects. This makes DataLoader `spawn` worker startup fast (workers inherit the dataset
via pickle).

### Per-worker LRU image cache (`_load_page_image`)

```python
@functools.lru_cache(maxsize=8)
def _load_page_image(path: str) -> np.ndarray | None:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```

A module-level function decorated with `lru_cache`. With `persistent_workers=True`,
each spawned worker process keeps its own cache alive across batches and epochs.
`cv2` is used instead of PIL to avoid libjpeg fork-mutex deadlocks.

### PageGroupSampler

```python
class PageGroupSampler(Sampler[int]):
    def __init__(self, path_idx, shuffle=True, seed=42): ...
    def set_epoch(self, epoch): ...          # re-shuffles pages each epoch
    def __iter__(self):
        # shuffle page order, then yield all sample indices from each page
        # consecutively
```

With random `DataLoader(shuffle=True)`, consecutive indices are from random pages.
The LRU cache sees only cache misses — every sample loads a fresh JPEG.

With `PageGroupSampler`, consecutive indices are from the **same page**. The cache
load pattern becomes:

```
page_A  →  load JPEG  →  cache hit  →  cache hit  →  cache hit  (×20 samples)
page_B  →  load JPEG  →  cache hit  →  ...
```

Result: **~20× fewer JPEG decodes per epoch** (one per page instead of one per sample).

`set_epoch(epoch)` changes the page shuffle order each epoch using `seed + epoch`, so
the training order changes every epoch despite the page-grouped structure.

Used only for visual training; geom-only training uses standard `shuffle=True` since
no image I/O occurs.

---

## 7. Training

```bash
# Geometry-only (recommended starting point)
sf-train-lps --epochs 30 --batch 16384 --workers 8

# Visual scorer
sf-train-lps --visual --epochs 30 --batch 1024 --workers 8
```

Full options:

```
--data-dir PATH       root with train/ val/ subdirs  [data/generated]
--output-dir PATH     checkpoint directory           [runs/lps/]
--epochs INT          [30]
--batch INT           batch size; 16384 for geom, 1024 for visual
--visual              train VisualScorer (GeomScorer is default)
--neg-per-pos INT     hard negatives per positive pair  [3]
--bbox-jitter FLOAT   coordinate noise fraction         [0.02]
--lr FLOAT            [1e-3]
--weight-decay FLOAT  [1e-4]
--workers INT         DataLoader workers                [8]
--device STR          [cuda]
--seed INT            [42]
```

### Optimiser / scheduler

```
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingLR(T_max=epochs)
BCEWithLogitsLoss(pos_weight=dataset.pos_weight())
```

### DataLoader configuration

| Setting | Geom-only | Visual |
|---|---|---|
| `multiprocessing_context` | `fork` | `spawn` |
| `persistent_workers` | True | True |
| `prefetch_factor` | 8 | 8 |
| `sampler` | `shuffle=True` | `PageGroupSampler` |

**Why fork for geom-only**: workers never open images or touch CUDA — fork is safe and
has zero startup cost. Spawn would re-import torch in every worker.

**Why spawn for visual**: avoids inheriting the parent process's CUDA context and any
open libjpeg state into workers. With compact numpy arrays, spawn pickling is fast.

### Output

```
runs/lps/
├── scorer_best.pt    # checkpoint with highest val accuracy
└── scorer_final.pt   # checkpoint at last epoch
```

Checkpoint format (loaded by `load_checkpoint()`):

```python
{
    "model_state_dict": ...,
    "model_type": "geom" | "visual",
    "epoch": int,
    "val_accuracy": float,
    "val_loss": float,
}
```

---

## 8. Evaluation

```bash
sf-eval-lps --weights runs/lps/scorer_best.pt --data-dir data/generated
```

Reports **page-level pair accuracy**: a page is correct only if all structure-label
assignments on it are correct. This is the operationally relevant metric — one wrong
pair on a page means the whole page extraction is wrong.

Compares `LearnedMatcher` vs `HungarianMatcher` side-by-side on the validation set.

---

## 9. Inference: LearnedMatcher

`structflo/cser/lps/matcher.py`

```python
from structflo.cser.lps import LearnedMatcher
from structflo.cser.pipeline import ChemPipeline

pipeline = ChemPipeline(
    matcher=LearnedMatcher("runs/lps/scorer_best.pt")
)
pairs = pipeline.process("page.jpg")
```

Constructor args:

| Arg | Default | Meaning |
|---|---|---|
| `weights` | required | Local `.pt` path, version tag `"v1.0"`, or `None` (auto-download latest) |
| `min_score` | `0.5` | Pairs below this score are left unmatched |
| `device` | `"cuda"` | Inference device |
| `geometry_only` | `False` | Skip visual CNN even if checkpoint has it |
| `max_dist_px` | `None` | Optional Euclidean pre-filter to skip obviously wrong pairs |

### Algorithm

```
1. Split detections → structures (class 0), labels (class 1)
2. Build all n×m candidate pairs
3. Batch score matrix in one forward pass
4. cost_matrix = 1.0 - score_matrix
5. linear_sum_assignment(cost_matrix)   ← Hungarian, same as before
6. Drop assignments where score < min_score
7. Return list[CompoundPair]
```

`match_distance` is set to `1.0 - score` (lower = more confident), keeping the same
semantics as `HungarianMatcher.match_distance` for downstream consumers.

### Page size inference

Used to normalise geometric features. Priority:

1. `image.shape` if image is passed
2. Bounding box extents of all detections
3. Fallback: A4@300DPI (2480×3508)

---

## 10. Package Structure

```
structflo/cser/lps/
├── __init__.py       exports LearnedMatcher
├── features.py       geom_features(), crop_region() — pure numpy, no torch
├── scorer.py         GeomScorer, VisualScorer, save_checkpoint, load_checkpoint
├── dataset.py        LPSDataset, PageGroupSampler, _load_page_image
├── train.py          sf-train-lps entry point
├── evaluate.py       sf-eval-lps entry point
└── matcher.py        LearnedMatcher(BaseMatcher)
```

Dependency directions (no cycles):

```
features.py ← dataset.py ← train.py
            ← matcher.py
scorer.py   ← train.py
            ← matcher.py
pipeline/*  ← matcher.py   (one-way: pipeline does not import lps)
```

---

## 11. Weights Registry

`structflo/cser/weights.py` contains a `"cser-lps"` entry following the same pattern
as `"cser-detector"`. `LearnedMatcher` calls `resolve_weights("cser-lps", version=...)`.

Publishing workflow (after training):

```bash
# 1. Create HF Hub repo
huggingface-cli repo create structflo-cser-lps --type model

# 2. Push checkpoint
huggingface-cli upload sidxz/structflo-cser-lps runs/lps/scorer_best.pt scorer_best.pt \
    --revision weights-v1.0

# 3. Compute sha256
sha256sum runs/lps/scorer_best.pt

# 4. Fill in weights.py REGISTRY["cser-lps"]["v1.0"]["sha256"]
# 5. Set LATEST["cser-lps"] = "v1.0"
```
