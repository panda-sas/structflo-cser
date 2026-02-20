# structflo-cser

YOLO11l-based detector for chemical structures and their compound label IDs in scientific documents.

Part of the **structflo** library. Import as:
```python
from structflo.cser.pipeline import ChemPipeline
```

**Detection target:** A single bounding box (`compound_panel`) enclosing the union of a rendered chemical structure and its nearby label ID (e.g. `CHEMBL12345`).

---

## Installation

```bash
uv pip install -e .
```

This installs all dependencies and registers the `sf-*` CLI commands on your PATH.

---

## Pipeline

```
1. Fetch SMILES          →  sf-fetch-smiles
2. Download distractors  →  sf-download-distractors   (optional but recommended)
3. Generate dataset      →  sf-generate
4. Visualize labels      →  sf-viz                    (optional QA check)
5. Train YOLO            →  sf-train
6. Run inference         →  sf-detect
7. Annotate real PDFs    →  sf-annotate               (optional)
```

---

## Commands

### 1. Fetch SMILES from ChEMBL

Extracts ~20 k small-molecule SMILES from a local [ChEMBL SQLite database](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/).

```bash
sf-fetch-smiles \
  --db chembl_35/chembl_35_sqlite/chembl_35.db \
  --output data/smiles/chembl_smiles.csv \
  --n 20000
```

Output: `data/smiles/chembl_smiles.csv`

---

### 2. Download distractor images

Downloads real photographs from [Lorem Picsum](https://picsum.photos/) to use as hard-negative distractors during page generation.

```bash
sf-download-distractors --out data/distractors --count 1000
```

---

### 3. Generate synthetic dataset

Generates document-like pages (A4 @ 300 DPI or slide format) containing chemical structures, compound labels, and distractor elements.

```bash
sf-generate \
  --smiles data/smiles/chembl_smiles.csv \
  --out data/generated \
  --num-train 2000 --num-val 400 \
  --fonts-dir data/fonts \
  --distractors-dir data/distractors \
  --dpi 96,144,200,300 \
  --workers 0
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--num-train` | 2000 | Number of training pages |
| `--num-val` | 200 | Number of validation pages |
| `--dpi` | `96,144,200,300` | DPI values randomly sampled per page |
| `--grayscale` / `--no-grayscale` | on | Convert pages to grayscale |
| `--workers` | 0 (all CPUs) | Parallel workers; use `1` to disable multiprocessing |

**Output structure:**
```
data/generated/
├── train/
│   ├── images/         (JPEG pages)
│   ├── labels/         (YOLO .txt — union bbox per compound panel)
│   └── ground_truth/   (JSON with split struct_bbox / label_bbox / smiles)
└── val/
    ├── images/
    ├── labels/
    └── ground_truth/
```

---

### 4. Visualize labels (QA)

Overlays YOLO bounding boxes on a random sample of generated pages.

```bash
sf-viz --split both --n 30 --out data/viz
```

Green boxes = `chemical_structure`, blue boxes = `compound_label`.

---

### 5. Train

Fine-tunes YOLO11l on the generated dataset.

```bash
sf-train --epochs 50 --imgsz 1280 --batch 8
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | `yolo11l.pt` | Pretrained backbone |
| `--imgsz` | 1280 | Training resolution |
| `--batch` | 8 | Batch size (safe for A6000 48 GB) |
| `--resume` | — | Path to `last.pt` to resume an interrupted run |

**Output:** `runs/labels_detect/yolo11l_panels/weights/best.pt`

---

### 6. Detect

Runs the trained detector on images using sliding-window tiling (1536 px tiles, 20 % overlap).

```bash
# Single image
sf-detect --image page.png

# Directory of images
sf-detect --image_dir data/real/images/ --out detections/

# With Hungarian pairing of structures → labels
sf-detect --image page.png --pair --max_dist 300
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--weights` | `runs/.../best.pt` | Model weights |
| `--conf` | 0.3 | Confidence threshold |
| `--tile_size` | 1536 | Tile size in pixels |
| `--no_tile` | off | Run on full image (skips tiling) |
| `--grayscale` | off | Convert to grayscale before detection |
| `--pair` | off | Hungarian match structures → labels |

---

### 7. Annotate real PDFs (optional)

Web-based annotation tool for creating ground truth from real PDF documents.

```bash
sf-annotate --out data/real --port 8000
# then open http://127.0.0.1:8000 in a browser
```

---

## Package layout

```
structflo/cser/              # importable package (from structflo.cser import ...)
├── _geometry.py             # shared bbox utilities (boxes_intersect, try_place_box)
├── config.py                # PageConfig dataclass + make_page_config()
├── data/
│   ├── smiles.py            # load_smiles(), fetch_smiles_from_chembl_sqlite()
│   └── distractor_images.py # load_distractor_images(), download_picsum()
├── rendering/
│   ├── chemistry.py         # render_structure(), place_structure()
│   └── text.py              # draw_rotated_text(), add_label_near_structure(), load_font()
├── distractors/
│   ├── charts.py            # bar / scatter / line / pie chart generators
│   ├── shapes.py            # geometric shapes, noise patches, gradients
│   └── text_elements.py     # prose blocks, captions, footnotes, arrows, tables
├── generation/
│   ├── page.py              # make_page(), make_negative_page(), apply_noise()
│   └── dataset.py           # generate_dataset(), save_sample(), CLI entry point
├── training/
│   └── trainer.py           # train(), CLI entry point
├── inference/
│   ├── tiling.py            # generate_tiles()
│   ├── nms.py               # nms()
│   ├── pairing.py           # pair_detections() via Hungarian matching
│   └── detector.py          # detect_tiled(), detect_full(), draw_boxes(), CLI
└── viz/
    └── labels.py            # visualize_split(), draw_boxes(), CLI entry point

annotate/                    # Flask annotation tool (unchanged)
config/
├── data.yaml                # YOLO dataset paths
└── pipeline.yaml
data/                        # data files (gitignored)
runs/                        # training checkpoints (gitignored)
```

---

## Data directory layout

```
data/
├── smiles/
│   └── chembl_smiles.csv    # ~20 k SMILES from ChEMBL
├── fonts/                   # TTF/OTF fonts for label rendering
├── distractors/             # ~1 k real photos (sf-download-distractors output)
├── generated/               # synthetic dataset (sf-generate output)
│   ├── train/
│   └── val/
└── real/                    # manually annotated real pages (sf-annotate output)
    ├── images/
    ├── labels/
    └── ground_truth/
```

---

## YOLO label format

Each `.txt` label file contains one line per annotated object:

```
<class_id> <cx> <cy> <w> <h>   (all normalised to [0, 1])
```

| class_id | name |
|----------|------|
| 0 | chemical_structure |
| 1 | compound_label |

Ground-truth JSON files in `ground_truth/` contain raw pixel coordinates plus `smiles` and `label_text` for downstream analysis.

---

## Key design decisions

- **Union bounding box** — each compound panel is annotated as the union of structure + label (1 class for YOLO). The GT JSON preserves the individual boxes.
- **No horizontal flips** — chemical handedness matters; `fliplr=0` is enforced during training.
- **15 % negative pages** — pages with no structures teach the model to output nothing for non-chemistry content.
- **Multi-DPI generation** — pages at {96, 144, 200, 300} DPI create scale variance, improving robustness to different scanning resolutions.
- **Tiled inference** — A4 pages (2480 × 3508 px) are tiled into 1536 px chunks with 20 % overlap to stay within GPU memory.
