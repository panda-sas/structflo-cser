# Learned Pair Scorer (LPS)

Reference documentation for `structflo/cser/lps/` — the learned structure-label
association module that replaces Euclidean Hungarian matching with a trained
visual scorer.

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
| **Angled labels** | Real documents have labels at various angles; distance is irrelevant to angle |

A learned scorer sees both objects simultaneously and learns the statistical regularity
of how layout algorithms place labels relative to structures — including direction,
scale, page position, and visual appearance.

The design **preserves Hungarian matching** — only the cost metric changes:

```
cost[i,j] = 1.0 - scorer(sᵢ, lⱼ)
```

---

## 2. What Was Rejected and Why

**YOLO keypoint prediction** (predict dx,dy offset from structure to label) is
fundamentally flawed: there is no information in a molecule's bounding box that tells
the model where the layout algorithm placed its label. The keypoint head degenerates
to learning the prior ("labels are usually below") or detecting nearby text via
receptive field. See `learned_matcher_plan.md` §3 for the full argument.

**Instance segmentation (Mask R-CNN)** still requires the RPN to propose the
union bbox (structure + whitespace + label), reintroducing the whitespace-center
problem at the proposal stage.

**Geometry-only scoring** (MLP on 14 geometric features, no images) works well for
clean synthetic data but cannot distinguish true compound IDs from distractor text
(captions, equations, annotations) or handle angled labels. Visual features are
necessary.

---

## 3. Feature Engineering

### 3.1 Geometric features — 14-d vector

`structflo/cser/lps/features.py:geom_features()`

| # | Feature | Formula | Why |
|---|---|---|---|
| 0 | `dx_norm` | `(l_cx − s_cx) / s_w` | Lateral offset, structure-size-normalised |
| 1 | `dy_norm` | `(l_cy − s_cy) / s_h` | Vertical offset, structure-size-normalised |
| 2 | `dist_norm` | `√(dx² + dy²)` | Scale-invariant distance |
| 3 | `angle_sin` | `sin(atan2(dy, dx))` | Smooth angular encoding (no ±π discontinuity) |
| 4 | `angle_cos` | `cos(atan2(dy, dx))` | — |
| 5 | `size_ratio` | `l_area / s_area` | Label is typically 3–8% of structure area |
| 6 | `label_aspect` | `l_w / l_h` | Compound IDs are wide; captions are very wide |
| 7 | `struct_aspect` | `s_w / s_h` | Structure shape |
| 8 | `struct_page_x` | `s_cx / W` | Horizontal page position |
| 9 | `struct_page_y` | `s_cy / H` | Vertical position; near bottom → label above |
| 10 | `label_page_x` | `l_cx / W` | — |
| 11 | `label_page_y` | `l_cy / H` | — |
| 12 | `struct_conf` | YOLO confidence | Detection quality |
| 13 | `label_conf` | YOLO confidence | Detection quality |

### 3.2 Visual features

Two grayscale crops extracted from the page image:

- **Structure crop**: `128×128` → `_EmbedCNN` → 128-d embedding
- **Label crop**: `64×96` (taller than the old 32×96 to handle rotated text) → `_EmbedCNN` → 128-d embedding

---

## 4. Model — PairScorer

`structflo/cser/lps/scorer.py`

### 4.1 CNN encoder — `_EmbedCNN`

Shared architecture (separate weights for struct/label branches):

```
Stem:    Conv(1→32, 5×5) → BN → GELU → MaxPool(/2)
Stage 1: ResBlock(32) → DownConv(32→64, stride=2)
Stage 2: ResBlock(64) → DownConv(64→128, stride=2)
Pool:    GlobalAvgPool ‖ GlobalMaxPool → concat 256-d
Proj:    Linear(256→out_dim) → LayerNorm → GELU
```

**`_ResBlock(ch)`**: Two 3×3 convs + BN, residual skip, SE channel attention, GELU.

**`_SEBlock(ch)`**: Squeeze-and-Excitation — global avg pool → MLP → sigmoid → channel scale.
Learns which feature maps are most informative for each input, helping reject distractors.

Spatial flow for each crop size:

| Crop | After stem | After stage1 | After stage2 |
|---|---|---|---|
| Struct 128×128 | 64×64 | 32×32 | 16×16 |
| Label  64×96  | 32×48 | 16×24 |  8×12 |

Design rationale for angled labels:
- **5×5 stem kernel**: captures more of a rotated glyph in the first layer vs 3×3
- **Residual connections**: stable gradient flow through 6+ layers
- **SE attention**: recalibrates channels per spatial input
- **Avg+max pooling**: max pool retains strongest local activation (useful for sparse rotated text that avg would dilute)

### 4.2 PairScorer head

```
struct_crop [1×128×128]  →  _EmbedCNN  →  128-d
label_crop  [1× 64× 96]  →  _EmbedCNN  →  128-d
geom_feats  [14-d]        →  Linear(14→64) + LayerNorm + GELU  →  64-d

concat [320-d]
    → Linear(320→256) + LayerNorm + GELU + Dropout(0.2)
    → Linear(256→128) + GELU + Dropout(0.1)
    → Linear(128→1)
    → raw logit  (sigmoid → association probability)
```

**Parameter count: ~1.16 M**

---

## 5. Training Data

### Source

```
data/generated/
├── train/
│   ├── images/           30,000 × A4@300DPI JPEGs
│   └── ground_truth/     30,000 × JSON
└── val/
    ├── images/            5,000 × JPEGs
    └── ground_truth/      5,000 × JSON
```

### Pair construction (`LPSDataset._build`)

**Positive pairs** — every `(struct_bbox[i], label_bbox[i])` where `label_bbox` is
not null. ~5 per page → ~150K train, ~25K val.

**Hard negatives** — for each positive `(sᵢ, lᵢ)`, pair `sᵢ` with the `neg_per_pos`
(default 3) spatially nearest wrong labels on the same page.

Ratio: 1 positive : 3 negatives → `pos_weight ≈ 3.0` in `BCEWithLogitsLoss`.

### Bbox jitter (train only)

Simulates YOLO localisation noise:

```python
coord += Uniform(-0.02 × side, +0.02 × side)
```

### Visual augmentation (train only)

Applied in `__getitem__` when `augment=True`:

| Branch | Rotation | Flip | Brightness |
|---|---|---|---|
| Structure | ±180° | horizontal (p=0.5) | ×Uniform(0.75, 1.25) |
| Label     | ±45°  | horizontal (p=0.5) | ×Uniform(0.75, 1.25) |

Structure uses full rotation (molecules are rotationally symmetric). Label uses ±45°
(real-world text is semi-upright, rarely more than 45° from horizontal).

---

## 6. DataLoader Design

### Why things are set up this way

| Setting | Value | Reason |
|---|---|---|
| `multiprocessing_context` | `spawn` | Avoids inheriting CUDA/libjpeg state |
| `persistent_workers` | `True` | Workers stay alive across epochs (critical for spawn: avoids re-importing torch each epoch; also keeps LRU cache alive) |
| `prefetch_factor` | 8 | Workers queue 8 batches ahead; GPU never waits |
| `sampler` | `PageGroupSampler` | Yields all ~20 samples from each page consecutively → LRU cache hit rate ~95% |

### `_load_page_image` LRU cache

```python
@functools.lru_cache(maxsize=8)
def _load_page_image(path: str) -> np.ndarray | None:
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```

Module-level with `lru_cache`. Each spawn worker has its own independent cache that
persists across batches and epochs (via `persistent_workers`).

With random shuffle: every sample loads a different JPEG → O(N_samples) decodes/epoch.
With `PageGroupSampler`: all samples from a page are consecutive → O(N_pages) decodes/epoch (**~20× less I/O**).

---

## 7. Training

```bash
sf-train-lps --epochs 30 --batch 1024 --workers 8
```

Full options:

```
--data-dir PATH       root with train/ val/ subdirs  [data/generated]
--output-dir PATH     checkpoint directory           [runs/lps/]
--epochs INT          [30]
--batch INT           batch size                     [1024]
--neg-per-pos INT     hard negatives per positive    [3]
--bbox-jitter FLOAT   coordinate noise fraction      [0.02]
--lr FLOAT            [1e-3]
--weight-decay FLOAT  [1e-4]
--workers INT         DataLoader workers             [8]
--device STR          [cuda]
--seed INT            [42]
```

### Optimiser

```
AdamW(lr=1e-3, weight_decay=1e-4)
CosineAnnealingLR(T_max=epochs)
BCEWithLogitsLoss(pos_weight=dataset.pos_weight())
```

### Output

```
runs/lps/
├── scorer_best.pt    # checkpoint with highest val accuracy
└── scorer_last.pt    # most recent epoch (overwritten each epoch; safe to resume after crash)
```

Checkpoint format:

```python
{
    "state_dict":    ...,
    "epoch":         int,
    "val_accuracy":  float,
    "val_loss":      float,
}
```

---

## 8. Evaluation

```bash
sf-eval-lps --weights runs/lps/scorer_best.pt
```

Reports **page-level pair accuracy**: a page is correct only if all pairs are correctly
matched. Compares `LearnedMatcher` vs `HungarianMatcher` side-by-side.

```
--weights PATH     required — path to scorer_best.pt
--data-dir PATH    [data/generated/val]
--device STR       [cuda]
--max-pages INT    limit to first N pages (quick sanity check)
```

---

## 9. Inference — LearnedMatcher

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
| `max_dist_px` | `None` | Pre-filter: skip obviously wrong pairs to save compute |

### Algorithm

```
1. Split detections → structures (class 0), labels (class 1)
2. Optional max_dist_px pre-filter
3. Build all n×m candidate pairs
4. Extract geom features + image crops for each pair
5. Batch forward pass through PairScorer → score matrix
6. cost_matrix = 1.0 - score_matrix
7. linear_sum_assignment(cost_matrix)   ← Hungarian
8. Drop assignments where score < min_score
9. Return list[CompoundPair] with match_distance = 1 - score
```

---

## 10. Package Structure

```
structflo/cser/lps/
├── __init__.py       exports LearnedMatcher
├── features.py       geom_features(), crop_region() — pure numpy, no torch
├── scorer.py         _SEBlock, _ResBlock, _EmbedCNN, PairScorer
│                     save_checkpoint(), load_checkpoint()
├── dataset.py        LPSDataset, PageGroupSampler, _load_page_image, _augment_crop
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

`structflo/cser/weights.py` contains a `"cser-lps"` entry. `LearnedMatcher` calls
`resolve_weights("cser-lps", version=...)`.

Publishing workflow (after training):

```bash
# 1. Push checkpoint to HuggingFace Hub
huggingface-cli upload sidxz/structflo-cser-lps runs/lps/scorer_best.pt scorer_best.pt \
    --revision weights-v1.0

# 2. Compute sha256
sha256sum runs/lps/scorer_best.pt

# 3. Fill REGISTRY["cser-lps"]["v1.0"]["sha256"] in weights.py
# 4. Set LATEST["cser-lps"] = "v1.0"
```
