# ChemLabelDetect v2: Structure–Label Extraction Pipeline

**Goal:** Given document images containing N chemical structures each with a nearby text label
(non-semantic IDs like `SACC0005`, `LGNIA2000`), detect all structures and labels, pair them,
extract SMILES via DECIMER, extract label text via constrained OCR.

**Hardware:** NVIDIA A6000 Ada (48 GB VRAM)

---

## 1. Architecture Overview

### 1.1 Architecture

A single YOLO model detects **compound panels** — bounding boxes that tightly enclose both the
chemical structure and its nearby label ID together. Each panel crop is then sent to two
purpose-built recognizers in parallel:

| Component | Model | Task | Why |
|-----------|-------|------|-----|
| **Panel detector** | YOLO11l | Detect union bbox of structure + label | One model, no pairing needed |
| **Structure recognizer** | DECIMER | Panel crop → SMILES | Ignores label text automatically |
| **Label recognizer** | PaddleOCR | Panel crop → label string | Small crop = one text element, trivial for OCR |

**Pipeline flow:**
```
Page image
  └─→ YOLO11 (compound panel detection)
        └─→ For each panel crop:
              ├─→ DECIMER → SMILES  (ignores label text)
              └─→ PaddleOCR → label string  (only one text item in frame)
```

**Why a union bbox?** The structure and its label are always spatially coupled — annotating them
together eliminates the pairing problem entirely. DECIMER is robust to label text appearing in
the same crop. PaddleOCR operating on a small, isolated crop with a single text element is far
more reliable than running on the full page and filtering with regex.

### 1.2 Inference: Tiling, Not Resizing

**Critical issue:** Training images from the synthetic generator are ~2480×3508 (A4 at 300 DPI).
Resizing to 1280px means labels originally ~80px wide become ~40px — below reliable detection
threshold for both YOLO and DBNet.

**Solution: Sliding window with overlap.**

```
Full page (2480 × 3508)
  ├─ Tile 0: [0:1536, 0:1536]         ─┐
  ├─ Tile 1: [0:1536, 1152:2688]       │  20% overlap
  ├─ Tile 2: [0:1536, 2304:3508]       │  between tiles
  ├─ Tile 3: [1152:2480, 0:1536]       │
  ├─ Tile 4: [1152:2480, 1152:2688]    │
  └─ Tile 5: [1152:2480, 2304:3508]   ─┘
       │
       ▼
  Run detector on each tile at native resolution
       │
       ▼
  Map tile-local boxes → page-global coordinates
       │
       ▼
  NMS merge (IoU > 0.5) to deduplicate overlapping detections
```

This applies to BOTH the structure detector and the text detector. The structure detector
can also run on the full page at 1280 as a coarse pass (structures are large enough),
but the text detector MUST tile.

---

## 2. Project Structure

```
chemlabeldetect/
├── config/
│   ├── data.yaml                       # YOLO dataset config
│   └── pipeline.yaml                   # Inference config (thresholds, tiling, etc.)
├── data/
│   ├── smiles/
│   │   └── chembl_smiles.csv
│   ├── fonts/
│   ├── distractors/                    # Distractor text corpus
│   │   ├── lorem_paragraphs.txt        # Fake body text blocks
│   │   └── caption_templates.txt       # "Figure N.", "Scheme N." etc.
│   └── generated/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── val/
│           ├── images/
│           └── labels/
├── scripts/
│   ├── 01_fetch_smiles.py
│   ├── 02_generate_dataset.py          # Synthetic compositor with distractors
│   ├── 03_train.py                     # YOLO11 fine-tuning (structures only)
│   ├── 04_pipeline.py                  # Full inference pipeline with tiling
│   ├── 05_evaluate.py
│   ├── 06_visualize.py
│   └── utils/
│       ├── tiling.py                   # Sliding window + NMS merge
│       ├── pairing.py                  # Multi-signal structure↔label matching
│       └── label_decode.py             # Constrained OCR decoding
├── runs/
├── environment.yml
└── README.md
```

---

## 3. Environment

```yaml
# environment.yml
name: chemlabeldetect
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - conda-forge::rdkit
  - pip:
    - ultralytics>=8.3.0
    - decimer>=2.6.0
    - paddlepaddle-gpu>=2.6.0       # GPU-accelerated PaddleOCR
    - paddleocr>=2.8.0              # DBNet text detection + recognition
    - chembl-webresource-client>=0.10.8
    - Pillow>=10.0
    - scipy>=1.11
    - numpy>=1.24
    - opencv-python-headless>=4.8
    - matplotlib>=3.7
    - tqdm>=4.65
```

```bash
conda env create -f environment.yml
conda activate chemlabeldetect
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "from paddleocr import PaddleOCR; print('PaddleOCR OK')"
```

---

## 4. Synthetic Dataset Generation

### 4.1 Design Principles

The synthetic generator must produce pages that look like REAL documents, not clean
structure-only canvases. Failure to include distractors is the #1 cause of domain gap.

**Required page elements:**
1. Chemical structures (target: detect these)
2. Label IDs near structures (target: detect + read these)
3. **Distractors — all of the following:**
   - Body text paragraphs (prose blocks between/around figures)
   - Figure captions ("Figure 1.", "Scheme 2.", "Table 3.")
   - Reaction arrows (→, ⇌, drawn as simple lines)
   - Panel borders (rectangles around figure sub-panels)
   - Page numbers, headers, footers
   - R-group tables (grid of text near structures)
   - Random small text fragments (journal names, DOIs, etc.)
4. **Layout variation:**
   - Single-column and two-column layouts
   - Structures in grid panels vs. scattered
   - Labels rotated ±5–15° (occasional 90° for vertical labels)

### 4.2 YOLO Training Classes

We train with **one class only**:

| Class ID | Name | Annotation |
|----------|------|------------|
| 0 | `compound_panel` | Union bbox of chemical structure + its label ID |

The union bbox is computed as `min/max` of the struct and label bounding boxes. For the ~10% of
structures that have no label, the annotation is just the structure bbox.

Distractors (prose, captions, arrows, tables, charts) appear in the image but get no annotation
— they are implicit negatives that teach YOLO not to fire on non-panel content.

### 4.3 `01_fetch_smiles.py`

```python
#!/usr/bin/env python3
"""Fetch diverse SMILES from ChEMBL."""

import csv
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from tqdm import tqdm


def fetch_chembl_smiles(output_path: str, n: int = 20000):
    molecule = new_client.molecule
    results = molecule.filter(
        molecule_properties__mw_freebase__gte=150,
        molecule_properties__mw_freebase__lte=900,
        molecule_type="Small molecule",
    ).only(['molecule_chembl_id', 'molecule_structures'])

    collected = []
    seen = set()

    for rec in tqdm(results, total=n, desc="Fetching"):
        if len(collected) >= n:
            break
        try:
            smi = rec['molecule_structures']['canonical_smiles']
        except (KeyError, TypeError):
            continue
        if '.' in smi:
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)

        collected.append({
            'chembl_id': rec['molecule_chembl_id'],
            'smiles': canonical,
            'num_atoms': mol.GetNumHeavyAtoms(),
        })

    with open(output_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['chembl_id', 'smiles', 'num_atoms'])
        w.writeheader()
        w.writerows(collected)

    print(f"Saved {len(collected)} to {output_path}")


if __name__ == "__main__":
    fetch_chembl_smiles("data/smiles/chembl_smiles.csv", n=20000)
```

**Offline alternative** (faster, no rate limits): Download ChEMBL SQLite from
`https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/`, then:

```sql
SELECT DISTINCT cs.canonical_smiles, md.chembl_id
FROM compound_structures cs
JOIN molecule_dictionary md ON cs.molregno = md.molregno
JOIN compound_properties cp ON cs.molregno = cp.molregno
WHERE cp.mw_freebase BETWEEN 150 AND 900
  AND cs.canonical_smiles NOT LIKE '%.%'
ORDER BY RANDOM() LIMIT 20000;
```

### 4.4 `02_generate_dataset.py`

```python
#!/usr/bin/env python3
"""
Synthetic page generator with distractors.

Generates document-like pages containing:
  - Chemical structures (annotated as class 0 for YOLO)
  - Label IDs near structures (NOT annotated — PaddleOCR detects these)
  - Distractor elements: prose, captions, arrows, panel borders, page numbers
    (NOT annotated — implicit negatives for YOLO)

Output: YOLO-format dataset (images/ + labels/ with only structure bboxes).
"""

import csv
import math
import random
import string
import os
from pathlib import Path
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PageConfig:
    # Page dimensions (A4 @ 300 DPI)
    page_w: int = 2480
    page_h: int = 3508
    margin: int = 180

    # Structure rendering
    struct_size_range: tuple = (280, 550)
    bond_width_range: tuple = (1.5, 3.0)
    atom_font_range: tuple = (14, 28)

    # Label rendering
    label_font_range: tuple = (18, 32)
    label_offset_range: tuple = (8, 45)
    label_rotation_prob: float = 0.15       # Probability of rotated label
    label_rotation_range: tuple = (-15, 15) # Degrees
    label_90deg_prob: float = 0.03          # Probability of 90° vertical label

    # Layout
    min_structures: int = 1
    max_structures: int = 6
    two_column_prob: float = 0.3            # Probability of 2-column layout
    grid_jitter: float = 0.12               # Fraction of cell size

    # Distractors
    prose_block_prob: float = 0.7           # Probability of adding prose blocks
    caption_prob: float = 0.6               # Probability of figure caption
    arrow_prob: float = 0.3                 # Probability of reaction arrows
    panel_border_prob: float = 0.25         # Probability of panel borders
    page_number_prob: float = 0.5
    rgroup_table_prob: float = 0.15         # Probability of R-group table
    stray_text_prob: float = 0.4            # Random small text fragments

    # Noise (post-processing)
    jpeg_artifact_prob: float = 0.35
    blur_prob: float = 0.25
    noise_prob: float = 0.15
    warp_prob: float = 0.10
    brightness_prob: float = 0.40


# ═══════════════════════════════════════════════════════════════════
# Label ID generators
# ═══════════════════════════════════════════════════════════════════

LABEL_STYLES = {
    "alpha_num": lambda: (
        ''.join(random.choices(string.ascii_uppercase, k=random.randint(3, 5)))
        + ''.join(random.choices(string.digits, k=random.randint(3, 5)))
    ),
    "compound_num": lambda: f"Compound {random.randint(1, 99)}{random.choice('abcdefg')}",
    "simple_num": lambda: f"{random.randint(1, 50)}{random.choice(['', 'a', 'b', 'c', 'd'])}",
    "cas_like": lambda: f"{random.randint(10, 9999)}-{random.randint(10, 99)}-{random.randint(0, 9)}",
    "internal_dash": lambda: (
        ''.join(random.choices(string.ascii_uppercase, k=2))
        + '-' + ''.join(random.choices(string.digits, k=random.randint(3, 6)))
    ),
    "prefix_num": lambda: (
        random.choice(["CPD", "MOL", "HIT", "REF", "STD", "LIB", "SCR"])
        + '-' + str(random.randint(1, 99999)).zfill(random.randint(3, 5))
    ),
}


def random_label() -> str:
    return random.choice(list(LABEL_STYLES.values()))()


# ═══════════════════════════════════════════════════════════════════
# Structure rendering (RDKit)
# ═══════════════════════════════════════════════════════════════════

def render_structure(smiles: str, size: int, cfg: PageConfig) -> Optional[Image.Image]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        return None

    d = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = d.drawOptions()
    opts.bondLineWidth = random.uniform(*cfg.bond_width_range)
    opts.minFontSize = random.randint(*cfg.atom_font_range)
    opts.maxFontSize = opts.minFontSize + 8
    opts.additionalAtomLabelPadding = random.uniform(0.05, 0.2)
    opts.rotate = random.uniform(0, 360)
    if random.random() < 0.3:
        opts.useBWAtomPalette()

    d.DrawMolecule(mol)
    d.FinishDrawing()

    img = Image.open(BytesIO(d.GetDrawingText())).convert("RGBA")
    arr = np.array(img)

    # Tight crop
    mask = (arr[:, :, 3] > 0) & (
        (arr[:, :, 0] < 250) | (arr[:, :, 1] < 250) | (arr[:, :, 2] < 250)
    )
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    pad = 12
    y1 = max(0, ys.min() - pad)
    x1 = max(0, xs.min() - pad)
    y2 = min(arr.shape[0], ys.max() + pad)
    x2 = min(arr.shape[1], xs.max() + pad)

    cropped = img.crop((x1, y1, x2, y2))
    bg = Image.new("RGB", cropped.size, (255, 255, 255))
    bg.paste(cropped, mask=cropped.split()[3])
    return bg


# ═══════════════════════════════════════════════════════════════════
# Text rendering helpers
# ═══════════════════════════════════════════════════════════════════

def render_text(text: str, font: ImageFont.FreeTypeFont,
                color: tuple = (0, 0, 0), rotation: float = 0) -> Image.Image:
    """Render text to a tight PIL image, optionally rotated."""
    dummy = Image.new("RGB", (1, 1))
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0] + 8, bbox[3] - bbox[1] + 6

    img = Image.new("RGBA", (tw, th), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((4, -bbox[1] + 3), text, fill=color + (255,), font=font)

    if abs(rotation) > 0.5:
        img = img.rotate(-rotation, expand=True, fillcolor=(255, 255, 255, 0))

    # Convert to RGB on white
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def get_fonts(dirs: list[str]) -> list[str]:
    """Find all .ttf files in given directories."""
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith('.ttf'):
                    paths.append(os.path.join(root, f))
    if not paths:
        raise RuntimeError("No .ttf fonts found")
    return paths


# ═══════════════════════════════════════════════════════════════════
# Distractor generators
# ═══════════════════════════════════════════════════════════════════

LOREM = (
    "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim "
    "veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea "
    "commodo consequat Duis aute irure dolor in reprehenderit in voluptate "
    "velit esse cillum dolore eu fugiat nulla pariatur"
).split()

CAPTIONS = [
    "Figure {n}.", "Figure {n}:", "Scheme {n}.", "Scheme {n}:",
    "Table {n}.", "Chart {n}.", "Fig. {n}.", "Fig. {n}:",
    "Figure {n}. Chemical structures of selected compounds.",
    "Scheme {n}. Synthesis of target compounds.",
    "Figure {n}. SAR exploration around the lead scaffold.",
]


def random_prose(n_words: int = 40) -> str:
    return ' '.join(random.choices(LOREM, k=n_words)).capitalize() + '.'


def random_caption() -> str:
    template = random.choice(CAPTIONS)
    return template.format(n=random.randint(1, 12))


def draw_arrow(draw: ImageDraw.Draw, x: int, y: int, length: int,
               direction: str = "right", width: int = 2, color=(0, 0, 0)):
    """Draw a simple reaction arrow."""
    if direction == "right":
        draw.line([(x, y), (x + length, y)], fill=color, width=width)
        draw.polygon([(x + length, y),
                      (x + length - 12, y - 6),
                      (x + length - 12, y + 6)], fill=color)
    elif direction == "down":
        draw.line([(x, y), (x, y + length)], fill=color, width=width)
        draw.polygon([(x, y + length),
                      (x - 6, y + length - 12),
                      (x + 6, y + length - 12)], fill=color)


# ═══════════════════════════════════════════════════════════════════
# Label placement (relative to structure)
# ═══════════════════════════════════════════════════════════════════

PLACEMENTS = [
    ("below",       0.45),
    ("right",       0.20),
    ("above",       0.15),
    ("left",        0.10),
    ("below_right", 0.10),
]


def place_label(struct_bbox, label_size, direction, offset, pw, ph):
    sx1, sy1, sx2, sy2 = struct_bbox
    lw, lh = label_size
    scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2

    positions = {
        "below":       (scx - lw / 2, sy2 + offset),
        "above":       (scx - lw / 2, sy1 - lh - offset),
        "right":       (sx2 + offset, scy - lh / 2),
        "left":        (sx1 - lw - offset, scy - lh / 2),
        "below_right": (sx2 + offset // 2, sy2 + offset // 2),
    }
    lx, ly = positions.get(direction, positions["below"])
    lx = max(0, min(int(lx), pw - lw))
    ly = max(0, min(int(ly), ph - lh))
    return lx, ly


# ═══════════════════════════════════════════════════════════════════
# Page compositor
# ═══════════════════════════════════════════════════════════════════

def generate_page(smiles_list: list[str], cfg: PageConfig,
                  font_paths: list[str]) -> Optional[tuple]:
    """
    Generate one synthetic page.

    Returns:
        (page_image, annotations)
        annotations: list of dicts, each with:
            struct_bbox: (x1, y1, x2, y2) in pixels — YOLO annotation
            label_bbox:  (x1, y1, x2, y2) in pixels — for GT evaluation only
            label_text:  str
            smiles:      str
    """
    n_struct = random.randint(cfg.min_structures, cfg.max_structures)
    chosen = random.sample(smiles_list, min(n_struct, len(smiles_list)))

    # Background
    bg = tuple(random.randint(240, 255) for _ in range(3))
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), bg)
    draw = ImageDraw.Draw(page)

    # ── Layout grid ──────────────────────────────────────────────
    n = len(chosen)
    two_col = random.random() < cfg.two_column_prob and n >= 2

    if two_col:
        col_w = (cfg.page_w - 2 * cfg.margin) // 2
        # Reserve top/bottom for prose
        top_margin = cfg.margin + random.randint(100, 400)
        usable_h = cfg.page_h - top_margin - cfg.margin
        cols = 2
        rows = math.ceil(n / 2)
    else:
        col_w = cfg.page_w - 2 * cfg.margin
        top_margin = cfg.margin + random.randint(50, 200)
        usable_h = cfg.page_h - top_margin - cfg.margin
        if n <= 2:
            cols, rows = n, 1
        elif n <= 4:
            cols, rows = 2, 2
        else:
            cols, rows = 3, 2
        col_w = (cfg.page_w - 2 * cfg.margin) // cols

    cell_h = usable_h // max(rows, 1)

    annotations = []

    for idx, smi in enumerate(chosen):
        col = idx % cols
        row = idx // cols

        cx = cfg.margin + col * col_w + col_w // 2
        cy = top_margin + row * cell_h + cell_h // 2
        cx += int(random.gauss(0, cfg.grid_jitter * col_w))
        cy += int(random.gauss(0, cfg.grid_jitter * cell_h))

        # Render structure
        sz = random.randint(*cfg.struct_size_range)
        struct_img = render_structure(smi, sz, cfg)
        if struct_img is None:
            continue

        sw, sh = struct_img.size
        sx = max(cfg.margin, min(cx - sw // 2, cfg.page_w - cfg.margin - sw))
        sy = max(top_margin, min(cy - sh // 2, cfg.page_h - cfg.margin - sh))
        page.paste(struct_img, (sx, sy))
        struct_bbox = (sx, sy, sx + sw, sy + sh)

        # Render label
        font_path = random.choice(font_paths)
        font_size = random.randint(*cfg.label_font_range)
        font = ImageFont.truetype(font_path, font_size)
        label_text = random_label()

        rotation = 0.0
        if random.random() < cfg.label_90deg_prob:
            rotation = 90.0
        elif random.random() < cfg.label_rotation_prob:
            rotation = random.uniform(*cfg.label_rotation_range)

        label_color = (0, 0, 0) if random.random() < 0.8 else random.choice([
            (50, 50, 50), (0, 0, 128), (80, 0, 0),
        ])
        label_img = render_text(label_text, font, label_color, rotation)
        lw, lh = label_img.size

        direction = random.choices(
            [d for d, _ in PLACEMENTS], [w for _, w in PLACEMENTS], k=1
        )[0]
        offset = random.randint(*cfg.label_offset_range)
        lx, ly = place_label(struct_bbox, (lw, lh), direction, offset,
                             cfg.page_w, cfg.page_h)
        page.paste(label_img, (lx, ly))
        label_bbox = (lx, ly, lx + lw, ly + lh)

        annotations.append({
            "struct_bbox": struct_bbox,
            "label_bbox": label_bbox,
            "label_text": label_text,
            "smiles": smi,
        })

    if not annotations:
        return None

    # ── Distractors (NOT annotated) ──────────────────────────────

    # 1. Prose blocks
    if random.random() < cfg.prose_block_prob:
        n_blocks = random.randint(1, 3)
        for _ in range(n_blocks):
            prose = random_prose(random.randint(20, 80))
            fp = random.choice(font_paths)
            fs = random.randint(10, 14)
            try:
                pfont = ImageFont.truetype(fp, fs)
            except Exception:
                continue
            # Wrap text manually (crude)
            max_chars = (cfg.page_w - 2 * cfg.margin) // (fs // 2)
            lines = [prose[i:i + max_chars] for i in range(0, len(prose), max_chars)]
            py = random.choice([
                random.randint(cfg.margin, top_margin - 20),          # Above structures
                random.randint(cfg.page_h - cfg.margin - 100, cfg.page_h - cfg.margin),  # Below
            ])
            px = cfg.margin + random.randint(0, 50)
            for line in lines[:4]:
                draw.text((px, py), line, fill=(30, 30, 30), font=pfont)
                py += fs + 4

    # 2. Figure caption
    if random.random() < cfg.caption_prob:
        cap = random_caption()
        fp = random.choice(font_paths)
        try:
            cfont = ImageFont.truetype(fp, random.randint(12, 16))
        except Exception:
            cfont = ImageFont.load_default()
        cap_y = max(a["struct_bbox"][3] for a in annotations) + random.randint(30, 80)
        if cap_y < cfg.page_h - 100:
            draw.text((cfg.margin, cap_y), cap, fill=(0, 0, 0), font=cfont)

    # 3. Reaction arrows
    if random.random() < cfg.arrow_prob:
        for _ in range(random.randint(1, 3)):
            ax = random.randint(cfg.margin, cfg.page_w - cfg.margin - 100)
            ay = random.randint(top_margin, cfg.page_h - cfg.margin)
            draw_arrow(draw, ax, ay,
                       random.randint(40, 120),
                       random.choice(["right", "down"]),
                       random.randint(1, 3))

    # 4. Panel borders
    if random.random() < cfg.panel_border_prob:
        bx1 = cfg.margin - 20
        by1 = top_margin - 20
        bx2 = cfg.page_w - cfg.margin + 20
        by2 = max(a["struct_bbox"][3] for a in annotations) + 60
        draw.rectangle([bx1, by1, bx2, min(by2, cfg.page_h - 50)],
                       outline=(0, 0, 0), width=random.randint(1, 2))

    # 5. Page number
    if random.random() < cfg.page_number_prob:
        pn = str(random.randint(1, 200))
        try:
            pnfont = ImageFont.truetype(random.choice(font_paths), 12)
        except Exception:
            pnfont = ImageFont.load_default()
        draw.text((cfg.page_w // 2 - 10, cfg.page_h - cfg.margin + 20),
                  pn, fill=(100, 100, 100), font=pnfont)

    # 6. Stray text fragments (DOIs, journal names, headers)
    if random.random() < cfg.stray_text_prob:
        fragments = [
            f"doi: 10.{random.randint(1000,9999)}/{''.join(random.choices(string.ascii_lowercase, k=8))}",
            random.choice(["J. Med. Chem.", "Bioorg. Med. Chem. Lett.", "J. Org. Chem.",
                           "Eur. J. Med. Chem.", "ACS Med. Chem. Lett."]),
            f"Vol. {random.randint(1, 120)}, No. {random.randint(1, 24)}, pp. {random.randint(100, 9999)}-{random.randint(100, 9999)}",
        ]
        for frag in random.sample(fragments, random.randint(1, 2)):
            try:
                ffont = ImageFont.truetype(random.choice(font_paths),
                                           random.randint(8, 11))
            except Exception:
                continue
            fx = random.randint(cfg.margin, cfg.page_w - 300)
            fy = random.choice([
                random.randint(20, cfg.margin - 10),
                random.randint(cfg.page_h - cfg.margin + 10, cfg.page_h - 20),
            ])
            draw.text((fx, fy), frag, fill=(120, 120, 120), font=ffont)

    # ── Post-processing noise ────────────────────────────────────
    page = apply_noise(page, cfg)

    return page, annotations


# ═══════════════════════════════════════════════════════════════════
# Document noise
# ═══════════════════════════════════════════════════════════════════

def apply_noise(page: Image.Image, cfg: PageConfig) -> Image.Image:
    img = np.array(page)

    if random.random() < cfg.blur_prob:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < cfg.jpeg_artifact_prob:
        q = random.randint(40, 85)
        _, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    if random.random() < cfg.noise_prob:
        d = random.uniform(0.001, 0.005)
        mask = np.random.random(img.shape[:2])
        img[mask < d / 2] = 0
        img[mask > 1 - d / 2] = 255

    if random.random() < cfg.warp_prob:
        h, w = img.shape[:2]
        off = random.randint(5, 20)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [random.randint(0, off), random.randint(0, off)],
            [w - random.randint(0, off), random.randint(0, off)],
            [w - random.randint(0, off), h - random.randint(0, off)],
            [random.randint(0, off), h - random.randint(0, off)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255))

    if random.random() < cfg.brightness_prob:
        a = random.uniform(0.85, 1.15)
        b = random.randint(-15, 15)
        img = np.clip(a * img + b, 0, 255).astype(np.uint8)

    return Image.fromarray(img)


# ═══════════════════════════════════════════════════════════════════
# YOLO format output
# ═══════════════════════════════════════════════════════════════════

def bbox_to_yolo(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    return (
        (x1 + x2) / 2 / img_w,
        (y1 + y2) / 2 / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h,
    )


def generate_dataset(smiles_csv, output_dir, n_train=2000, n_val=400,
                     cfg=None):
    if cfg is None:
        cfg = PageConfig()

    with open(smiles_csv) as f:
        smiles_list = [r['smiles'] for r in csv.DictReader(f)]
    print(f"Loaded {len(smiles_list)} SMILES")

    font_paths = get_fonts([
        "data/fonts/",
        "/usr/share/fonts/truetype/",
        "/usr/share/fonts/truetype/dejavu/",
        "/usr/share/fonts/truetype/liberation/",
    ])
    print(f"Found {len(font_paths)} fonts")

    for split, n_pages in [("train", n_train), ("val", n_val)]:
        img_dir = Path(output_dir) / split / "images"
        lbl_dir = Path(output_dir) / split / "labels"
        # Also save full GT (including label bboxes) for evaluation
        gt_dir = Path(output_dir) / split / "ground_truth"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        done = 0
        with tqdm(total=n_pages, desc=f"Generating {split}") as pbar:
            while done < n_pages:
                result = generate_page(smiles_list, cfg, font_paths)
                if result is None:
                    continue

                page_img, annotations = result
                pid = f"{split}_{done:06d}"

                page_img.save(str(img_dir / f"{pid}.png"), "PNG")

                # YOLO label file: structure bboxes ONLY (class 0)
                with open(lbl_dir / f"{pid}.txt", 'w') as f:
                    for ann in annotations:
                        yb = bbox_to_yolo(ann["struct_bbox"], cfg.page_w, cfg.page_h)
                        f.write(f"0 {yb[0]:.6f} {yb[1]:.6f} {yb[2]:.6f} {yb[3]:.6f}\n")

                # Full GT JSON (for pairing evaluation)
                import json
                with open(gt_dir / f"{pid}.json", 'w') as f:
                    serializable = []
                    for ann in annotations:
                        serializable.append({
                            "struct_bbox": list(ann["struct_bbox"]),
                            "label_bbox": list(ann["label_bbox"]),
                            "label_text": ann["label_text"],
                            "smiles": ann["smiles"],
                        })
                    json.dump(serializable, f)

                done += 1
                pbar.update(1)


if __name__ == "__main__":
    generate_dataset(
        smiles_csv="data/smiles/chembl_smiles.csv",
        output_dir="data/generated",
        n_train=2000,
        n_val=400,
    )
```

---

## 5. YOLO Dataset Config

```yaml
# config/data.yaml
path: data/generated
train: train
val: val

nc: 1
names:
  0: compound_panel
```

---

## 6. Training: `03_train.py`

```python
#!/usr/bin/env python3
"""Fine-tune YOLO11 for chemical structure detection only."""

from ultralytics import YOLO


def train():
    model = YOLO("yolo11l.pt")
    # Experimental: swap to "yolo26l.pt" for STAL small-target benefits

    results = model.train(
        data="config/data.yaml",

        epochs=150,
        patience=30,
        batch=16,
        imgsz=1280,              # Structures are large — 1280 is fine

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        cos_lr=True,

        # Document-specific augmentation
        augment=True,
        hsv_h=0.005,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.1,
        scale=0.3,
        shear=1.0,
        flipud=0.0,             # NEVER flip
        fliplr=0.0,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,

        project="runs/chemlabeldetect",
        name="yolo11l_panels",
        workers=8,
        seed=42,
        plots=True,
        save_period=25,
    )

    metrics = model.val()
    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    train()
```

**Target:** mAP50 > 0.95 for `chemical_structure` (single class, visually distinctive — should be easy).

---

## 7. Tiling: `utils/tiling.py`

```python
"""Sliding-window tiling with NMS merge for high-res document pages."""

import numpy as np
from dataclasses import dataclass


@dataclass
class TileConfig:
    tile_size: int = 1536       # Tile width/height in pixels
    overlap: float = 0.20       # Overlap fraction between adjacent tiles
    nms_iou_thresh: float = 0.50


def generate_tiles(img_w: int, img_h: int, cfg: TileConfig) -> list[tuple]:
    """
    Generate tile coordinates as (x_start, y_start, x_end, y_end).
    Tiles may extend beyond image bounds — caller should pad or clamp.
    """
    step = int(cfg.tile_size * (1 - cfg.overlap))
    tiles = []

    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x_end = min(x + cfg.tile_size, img_w)
            y_end = min(y + cfg.tile_size, img_h)
            # Adjust start to maintain tile_size when hitting edge
            x_start = max(0, x_end - cfg.tile_size)
            y_start = max(0, y_end - cfg.tile_size)
            tiles.append((x_start, y_start, x_end, y_end))
            if x_end >= img_w:
                break
            x += step
        if y_end >= img_h:
            break
        y += step

    return tiles


def tile_local_to_global(boxes: np.ndarray, tile_origin: tuple) -> np.ndarray:
    """Shift [x1,y1,x2,y2] boxes from tile-local to page-global coords."""
    ox, oy = tile_origin
    shifted = boxes.copy()
    shifted[:, [0, 2]] += ox
    shifted[:, [1, 3]] += oy
    return shifted


def nms_merge(boxes: np.ndarray, scores: np.ndarray,
              iou_thresh: float = 0.5) -> np.ndarray:
    """Standard NMS. Returns indices to keep."""
    if len(boxes) == 0:
        return np.array([], dtype=int)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        remaining = np.where(iou <= iou_thresh)[0]
        order = order[remaining + 1]

    return np.array(keep)
```

---

## 8. ~~Pairing: `utils/pairing.py`~~ — Not needed

Structure↔label pairing is no longer required. YOLO detects the union bbox of each
structure+label panel, so the two are already coupled at detection time. The `utils/pairing.py`
module and `scipy` dependency can be removed.

---

## 9. Constrained OCR: `utils/label_decode.py`

```python
"""
Constrained OCR decoding for chemical compound IDs.

Since labels are non-semantic IDs with predictable patterns, we apply:
  1. Charset restriction (allowlist)
  2. Confusion pair correction (O↔0, I↔1, S↔5, B↔8, Z↔2)
  3. Regex-based pattern validation
  4. Confidence-based ambiguity resolution
"""

import re
from dataclasses import dataclass, field


@dataclass
class LabelDecoderConfig:
    # Allowed characters in your label IDs
    allowed_chars: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-/"
    # If True, strip any character not in allowed_chars
    strict_charset: bool = True
    # Known regex patterns for valid labels (in order of priority)
    valid_patterns: list = field(default_factory=lambda: [
        r'^[A-Z]{2,6}[0-9]{2,6}$',           # SACC0005, LGNIA2000
        r'^[A-Z]{2,4}-[0-9]{2,6}$',           # CPD-00123, MO-4521
        r'^Compound\s+\d{1,3}[a-g]?$',        # Compound 3a
        r'^\d{1,3}[a-g]?$',                   # 1, 3a, 14b
        r'^\d{2,5}-\d{2}-\d$',                # CAS-like: 123-45-6
        r'^[A-Z]{2,4}-\d{2,6}$',              # XX-1234
    ])


# Common OCR confusion pairs: (wrong_char, correct_char, context)
CONFUSION_PAIRS = [
    ('O', '0', 'in_digit_run'),     # O → 0 when surrounded by digits
    ('0', 'O', 'in_alpha_run'),     # 0 → O when surrounded by letters
    ('I', '1', 'in_digit_run'),
    ('1', 'I', 'in_alpha_run'),
    ('l', '1', 'in_digit_run'),     # lowercase l → 1
    ('S', '5', 'in_digit_run'),
    ('5', 'S', 'in_alpha_run'),
    ('B', '8', 'in_digit_run'),
    ('Z', '2', 'in_digit_run'),
    ('G', '6', 'in_digit_run'),
]


def fix_confusions(text: str) -> str:
    """
    Apply context-aware confusion fixes.

    Strategy: Identify alpha-runs and digit-runs, then fix characters
    that don't belong in their local context.
    """
    if len(text) < 2:
        return text

    result = list(text)

    for i, ch in enumerate(result):
        # Look at neighbors to determine context
        left = result[i - 1] if i > 0 else ''
        right = result[i + 1] if i < len(result) - 1 else ''

        left_is_digit = left.isdigit()
        right_is_digit = right.isdigit()
        left_is_alpha = left.isalpha()
        right_is_alpha = right.isalpha()

        for wrong, correct, context in CONFUSION_PAIRS:
            if ch != wrong:
                continue
            if context == 'in_digit_run' and (left_is_digit or right_is_digit):
                result[i] = correct
                break
            if context == 'in_alpha_run' and (left_is_alpha or right_is_alpha):
                result[i] = correct
                break

    return ''.join(result)


def decode_label(raw_text: str, cfg: LabelDecoderConfig = None) -> dict:
    """
    Post-process OCR output for a label crop.

    Returns:
        {
            "text": cleaned label string,
            "confidence": float (0-1, how well it matches expected patterns),
            "pattern_matched": str or None (which regex matched),
        }
    """
    if cfg is None:
        cfg = LabelDecoderConfig()

    text = raw_text.strip()

    # Step 1: Uppercase
    text = text.upper()

    # Step 2: Fix confusions
    text = fix_confusions(text)

    # Step 3: Strip disallowed characters
    if cfg.strict_charset:
        text = ''.join(c for c in text if c in cfg.allowed_chars)

    # Step 4: Pattern matching
    matched_pattern = None
    for pattern in cfg.valid_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            matched_pattern = pattern
            break

    # Step 5: Confidence
    confidence = 0.9 if matched_pattern else 0.5
    if len(text) < 2:
        confidence = 0.1

    return {
        "text": text,
        "confidence": confidence,
        "pattern_matched": matched_pattern,
    }
```

---

## 10. Full Pipeline: `04_pipeline.py`

```python
#!/usr/bin/env python3
"""
Full inference pipeline:
  1. Tile page
  2. Run YOLO11 on tiles → compound panel boxes (struct+label union, merge via NMS)
  3. For each panel crop:
     a. DECIMER → SMILES  (ignores label text automatically)
     b. PaddleOCR → label string  (only one text item in the small crop)

Usage:
  python scripts/04_pipeline.py --image page.png
  python scripts/04_pipeline.py --image_dir /path/to/pages/ --output results.json
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR
from tqdm import tqdm

from utils.tiling import TileConfig, generate_tiles, tile_local_to_global, nms_merge


class ChemLabelPipeline:

    def __init__(
        self,
        yolo_weights: str = "runs/chemlabeldetect/yolo11l_panels/weights/best.pt",
        tile_cfg: TileConfig = None,
        yolo_conf: float = 0.5,
    ):
        self.yolo = YOLO(yolo_weights)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)
        self.tile_cfg = tile_cfg or TileConfig()
        self.yolo_conf = yolo_conf

    def _detect_panels_tiled(self, img: np.ndarray) -> list[dict]:
        """Run YOLO on tiles, merge panel boxes via NMS."""
        h, w = img.shape[:2]
        tiles = generate_tiles(w, h, self.tile_cfg)

        all_boxes, all_scores = [], []
        for (x1, y1, x2, y2) in tiles:
            tile = img[y1:y2, x1:x2]
            results = self.yolo(tile, imgsz=self.tile_cfg.tile_size,
                                conf=self.yolo_conf, verbose=False)[0]
            for box in results.boxes:
                local_bbox = box.xyxy[0].cpu().numpy()
                global_bbox = tile_local_to_global(local_bbox.reshape(1, 4), (x1, y1))[0]
                all_boxes.append(global_bbox)
                all_scores.append(float(box.conf[0]))

        if not all_boxes:
            return []

        boxes_arr = np.array(all_boxes)
        scores_arr = np.array(all_scores)
        keep = nms_merge(boxes_arr, scores_arr, self.tile_cfg.nms_iou_thresh)
        return [{"bbox": boxes_arr[i], "conf": scores_arr[i]} for i in keep]

    def _extract_smiles(self, crop: Image.Image) -> str:
        from DECIMER import predict_SMILES
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            crop.save(f.name)
            try:
                return predict_SMILES(f.name)
            finally:
                os.unlink(f.name)

    def _extract_label(self, crop: Image.Image) -> tuple[str | None, float]:
        """Run PaddleOCR on a small panel crop; return (text, confidence)."""
        import numpy as np
        result = self.ocr.ocr(np.array(crop), cls=True)
        if not result or not result[0]:
            return None, 0.0
        # Pick the highest-confidence text line
        best_text, best_conf = "", 0.0
        for line in result[0]:
            text, conf = line[1]
            if conf > best_conf:
                best_text, best_conf = text, conf
        return best_text or None, best_conf

    def process(self, image_path: str) -> list[dict]:
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)

        panels = self._detect_panels_tiled(img_np)

        results = []
        for p in panels:
            bb = p["bbox"]
            pad = 10
            crop_box = (
                max(0, int(bb[0]) - pad), max(0, int(bb[1]) - pad),
                min(img_pil.width, int(bb[2]) + pad),
                min(img_pil.height, int(bb[3]) + pad),
            )
            crop = img_pil.crop(crop_box)

            smiles = self._extract_smiles(crop)
            label_text, label_conf = self._extract_label(crop)

            results.append({
                "smiles": smiles,
                "label": label_text,
                "label_conf": round(label_conf, 3),
                "panel_bbox": bb.tolist(),
                "panel_conf": round(p["conf"], 3),
            })

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Single image path")
    parser.add_argument("--image_dir", help="Directory of images")
    parser.add_argument("--weights",
                        default="runs/chemlabeldetect/yolo11l_panels/weights/best.pt")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    pipeline = ChemLabelPipeline(yolo_weights=args.weights)

    all_results = {}
    paths = [Path(args.image)] if args.image else sorted(Path(args.image_dir).glob("*.png"))

    for p in tqdm(paths):
        try:
            all_results[p.name] = pipeline.process(str(p))
        except Exception as e:
            all_results[p.name] = {"error": str(e)}

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
```

---

## 11. Execution Order

```bash
# ── Setup ─────────────────────────────────────────────────────────
conda env create -f environment.yml
conda activate chemlabeldetect

# ── Step 1: Fetch SMILES ─────────────────────────────────────────
python scripts/01_fetch_smiles.py
# Output: data/smiles/chembl_smiles.csv  (~20K SMILES)

# ── Step 2: Generate synthetic dataset ───────────────────────────
python scripts/02_generate_dataset.py
# Output: data/generated/train/images/  (~2000 PNGs)
#         data/generated/train/labels/  (YOLO .txt, structures only)
#         data/generated/train/ground_truth/  (full JSON including labels)
#         data/generated/val/...

# ── Step 3: Spot-check data ──────────────────────────────────────
# Open a few generated images. Verify:
#   - YOLO boxes (visualize_labels.py) cover both structure AND label together
#   - ground_truth/*.json shows correct union_bbox, struct_bbox, label_bbox, label_text
#   - Distractor text, arrows, captions are present but have no annotation
python scripts/visualize_labels.py --split val --n 20 --out data/viz

# ── Step 4: Train YOLO11 (compound panel detection) ──────────────
python scripts/03_train.py
# ~45 min on A6000. Monitor: runs/chemlabeldetect/yolo11l_panels/
# Target: mAP50 > 0.95

# ── Step 5: Test pipeline on synthetic val images ────────────────
python scripts/04_pipeline.py \
  --image_dir data/generated/val/images/ \
  --weights runs/chemlabeldetect/yolo11l_panels/weights/best.pt \
  --output val_results.json

# ── Step 6: Test on REAL document images ─────────────────────────
python scripts/04_pipeline.py \
  --image /path/to/real/document_page.png \
  --output real_results.json

# ── Step 7: Evaluate ─────────────────────────────────────────────
python scripts/05_evaluate.py \
  --predictions val_results.json \
  --ground_truth data/generated/val/ground_truth/
```

---

## 12. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Panel mAP50 < 0.90 | Too few distractors, model fires on prose/tables | Add more prose blocks, increase `stray_text_prob` |
| Panel box cuts off label | Label placed far from structure, union box clipped | Check `label_offset_range`; verify `clamp_box` in `save_sample` |
| OCR reads wrong label from crop | Multiple text items in panel crop (label far from struct) | Tighten `label_offset_range` in `PageConfig` so label stays close |
| Good on synthetic, bad on real | Domain gap | Mix 50–100 annotated real pages into training; increase noise parameters |
| DECIMER invalid SMILES | Unusual structure or low crop quality | Binarize crops before DECIMER; check crop padding |
| Slow inference | DECIMER is bottleneck (~1-2s per structure) | Batch structure crops; or use MolScribe (faster) |
| OOM during training | Batch too large at imgsz=1280 | Reduce batch to 8; or use yolo11m instead of yolo11l |

---

## 13. Future Work

1. **Connector detection (high value):** Add `connector` class to YOLO (brackets, lines linking
   structure to label). Use connector endpoints for pairing instead of proximity. This is the
   single biggest upgrade for dense/complex layouts.

2. **Fine-tuned text recognizer:** Train PARSeq or TrOCR on synthetically generated label IDs
   with your exact charset and patterns. Expect near-perfect character accuracy.

3. **OBB for rotated labels:** If many labels are rotated, switch YOLO task to OBB
   (`yolo11l-obb.pt`) and train with oriented bounding boxes for structures. PaddleOCR already
   handles rotated text natively.

4. **Real-data mixing:** After the synthetic-only model works, annotate 50–100 real pages
   (using model-assisted pre-annotation) and mix into training for domain adaptation.

5. **End-to-end evaluation harness:** Canonical SMILES comparison (via RDKit) + exact label
   string match → single "fully correct" metric per page.
