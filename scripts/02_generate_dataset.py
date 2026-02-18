#!/usr/bin/env python3
"""
Synthetic page generator with distractors.

Generates document-like pages containing:
  - Chemical structures (annotated as class 0 for YOLO)
  - Label IDs near structures (not annotated)
  - Distractor elements: prose, captions, arrows, panel borders, page numbers

Output: YOLO-format dataset (images/ + labels/ with only structure bboxes).
"""

import argparse
import csv
import math
import os
import random
import string
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from tqdm import tqdm


@dataclass
class PageConfig:
    # Page dimensions (A4 @ 300 DPI)
    page_w: int = 2480
    page_h: int = 3508
    margin: int = 180

    # Structure rendering
    struct_size_range: Tuple[int, int] = (280, 550)
    bond_width_range: Tuple[float, float] = (1.5, 3.0)
    atom_font_range: Tuple[int, int] = (14, 28)

    # Label rendering
    label_font_range: Tuple[int, int] = (12, 36)
    label_offset_range: Tuple[int, int] = (8, 20)
    label_rotation_prob: float = 0.15
    label_rotation_range: Tuple[int, int] = (-15, 15)
    label_90deg_prob: float = 0.03

    # Layout
    min_structures: int = 1
    max_structures: int = 6
    two_column_prob: float = 0.3
    grid_jitter: float = 0.12

    # Distractors
    prose_block_prob: float = 0.7
    caption_prob: float = 0.6
    arrow_prob: float = 0.3
    panel_border_prob: float = 0.25
    page_number_prob: float = 0.5
    rgroup_table_prob: float = 0.15
    stray_text_prob: float = 0.4

    # Noise
    jpeg_artifact_prob: float = 0.35
    blur_prob: float = 0.25
    noise_prob: float = 0.15
    brightness_prob: float = 0.40


LABEL_STYLES = {
    "alpha_num": lambda: (
        "".join(random.choices(string.ascii_uppercase, k=random.randint(3, 5)))
        + "".join(random.choices(string.digits, k=random.randint(3, 5)))
    ),
    "compound_num": lambda: f"Compound {random.randint(1, 99)}{random.choice('abcdefg')}",
    "simple_num": lambda: f"{random.randint(1, 50)}{random.choice(['', 'a', 'b', 'c', 'd'])}",
    "cas_like": lambda: f"{random.randint(10, 9999)}-{random.randint(10, 99)}-{random.randint(0, 9)}",
    "internal_dash": lambda: (
        "".join(random.choices(string.ascii_uppercase, k=2))
        + "-" + "".join(random.choices(string.digits, k=random.randint(3, 6)))
    ),
    "prefix_num": lambda: (
        random.choice(["CPD", "MOL", "HIT", "REF", "STD", "LIB", "SCR"])
        + "-" + str(random.randint(1, 99999)).zfill(random.randint(3, 5))
    ),
}


def random_label() -> str:
    return random.choice(list(LABEL_STYLES.values()))()


def load_smiles(csv_path: Path) -> List[str]:
    smiles = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("smiles")
            if smi:
                smiles.append(smi)
    if not smiles:
        raise ValueError(f"No SMILES found in {csv_path}")
    return smiles


def render_structure(smiles: str, size: int, cfg: PageConfig) -> Optional[Image.Image]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        return None

    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = random.uniform(*cfg.bond_width_range)
    opts.minFontSize = random.randint(*cfg.atom_font_range)
    opts.maxFontSize = opts.minFontSize + 8
    opts.additionalAtomLabelPadding = random.uniform(0.05, 0.2)
    opts.rotate = random.uniform(0, 360)
    if random.random() < 0.3:
        opts.useBWAtomPalette()

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
    arr = np.array(img)

    mask = (arr[:, :, 3] > 0) & (
        (arr[:, :, 0] < 250) | (arr[:, :, 1] < 250) | (arr[:, :, 2] < 250)
    )
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
    return cropped


def load_font(font_paths: List[Path], size: int) -> ImageFont.ImageFont:
    if font_paths:
        for path in font_paths:
            try:
                return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def draw_rotated_text(
    base: Image.Image,
    text: str,
    position: Tuple[int, int],
    font: ImageFont.ImageFont,
    angle: float,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[int, int, int, int]:
    draw = ImageDraw.Draw(base)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    text_img = Image.new("RGBA", (text_w + 8, text_h + 8), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((4, 4), text, font=font, fill=fill)

    rotated = text_img.rotate(angle, expand=True)
    base.paste(rotated, position, rotated)

    x0, y0 = position
    x1 = x0 + rotated.size[0]
    y1 = y0 + rotated.size[1]
    return (x0, y0, x1, y1)


def clamp_box(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(1, min(x1, w))
    y1 = max(1, min(y1, h))
    return x0, y0, x1, y1


def boxes_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def place_structure(
    page: Image.Image,
    struct_img: Image.Image,
    cfg: PageConfig,
    existing_boxes: List[Tuple[int, int, int, int]],
    max_tries: int = 80,
) -> Optional[Tuple[int, int, int, int]]:
    w, h = page.size
    sw, sh = struct_img.size

    for _ in range(max_tries):
        x = random.randint(cfg.margin, w - cfg.margin - sw)
        y = random.randint(cfg.margin, h - cfg.margin - sh)
        box = (x, y, x + sw, y + sh)
        padded = (x - 6, y - 6, x + sw + 6, y + sh + 6)

        if any(boxes_intersect(padded, b) for b in existing_boxes):
            continue

        page.paste(struct_img, (x, y), struct_img)
        return box
    return None


def add_label_near_structure(
    page: Image.Image,
    struct_box: Tuple[int, int, int, int],
    cfg: PageConfig,
    font_paths: List[Path],
) -> Tuple[int, int, int, int]:
    w, h = page.size
    label = random_label()
    font_size = random.randint(*cfg.label_font_range)
    font = load_font(font_paths, font_size)

    x0, y0, x1, y1 = struct_box

    draw = ImageDraw.Draw(page)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    # Estimate max rotated dimensions (diagonal for worst case)
    max_rotated_dim = int(math.sqrt(text_w**2 + text_h**2)) + 16

    # 80% chance: place label below and centered
    if random.random() < 0.8:
        pos_x = x0 + (x1 - x0 - text_w) // 2  # Center horizontally
        pos_y = y1 + random.randint(*cfg.label_offset_range)  # Below structure
        angle = 0.0  # No rotation for centered-below labels
    else:
        # 20% chance: random placement around structure (original behavior)
        directions = [
            (0, -1),
            (1, 0),
            (0, 1),
            (-1, 0),
            (1, -1),
            (-1, -1),
            (1, 1),
            (-1, 1),
        ]
        dx, dy = random.choice(directions)
        offset = random.randint(*cfg.label_offset_range)

        base_x = x0 + (x1 - x0 - text_w) // 2
        base_y = y0 + (y1 - y0 - text_h) // 2

        pos_x = base_x + dx * (offset + (x1 - x0) // 2)
        pos_y = base_y + dy * (offset + (y1 - y0) // 2)

        angle = 0.0
        if random.random() < cfg.label_90deg_prob:
            angle = 90.0
        elif random.random() < cfg.label_rotation_prob:
            angle = random.uniform(*cfg.label_rotation_range)

    # Clamp position with safety margin accounting for potential rotation expansion
    safety_margin = max_rotated_dim if angle != 0.0 else text_w
    pos_x = max(cfg.margin, min(pos_x, w - cfg.margin - safety_margin))
    pos_y = max(cfg.margin, min(pos_y, h - cfg.margin - text_h))

    return draw_rotated_text(page, label, (pos_x, pos_y), font, angle)


def add_prose_block(page: Image.Image, cfg: PageConfig, font_paths: List[Path]) -> None:
    if random.random() > cfg.prose_block_prob:
        return

    w, h = page.size
    font = load_font(font_paths, random.randint(14, 18))
    draw = ImageDraw.Draw(page)
    block_w = random.randint(w // 4, w // 2)
    block_h = random.randint(80, 200)
    x0 = random.randint(cfg.margin, w - cfg.margin - block_w)
    y0 = random.randint(cfg.margin, h - cfg.margin - block_h)

    lines = []
    for _ in range(block_h // 18):
        words = [random.choice(PROSE_WORDS) for _ in range(random.randint(6, 12))]
        lines.append(" ".join(words))

    for i, line in enumerate(lines):
        draw.text((x0, y0 + i * 18), line, font=font, fill=(0, 0, 0))


def add_caption(page: Image.Image, cfg: PageConfig, font_paths: List[Path]) -> None:
    if random.random() > cfg.caption_prob:
        return

    w, h = page.size
    font = load_font(font_paths, random.randint(16, 20))
    draw = ImageDraw.Draw(page)
    cap = f"Figure {random.randint(1, 12)}. {random.choice(CAPTION_TEMPLATES)}"

    text_bbox = draw.textbbox((0, 0), cap, font=font)
    text_w = text_bbox[2] - text_bbox[0]

    x0 = random.randint(cfg.margin, w - cfg.margin - text_w)
    y0 = random.randint(cfg.margin, h - cfg.margin - 40)
    draw.text((x0, y0), cap, font=font, fill=(0, 0, 0))


def add_arrow(page: Image.Image, cfg: PageConfig) -> None:
    if random.random() > cfg.arrow_prob:
        return

    draw = ImageDraw.Draw(page)
    w, h = page.size
    x0 = random.randint(cfg.margin, w - cfg.margin - 120)
    y0 = random.randint(cfg.margin, h - cfg.margin - 40)
    length = random.randint(60, 140)

    x1 = x0 + length
    y1 = y0
    draw.line((x0, y0, x1, y1), fill=(0, 0, 0), width=2)
    draw.line((x1, y1, x1 - 10, y1 - 6), fill=(0, 0, 0), width=2)
    draw.line((x1, y1, x1 - 10, y1 + 6), fill=(0, 0, 0), width=2)


def add_panel_border(page: Image.Image, cfg: PageConfig) -> None:
    if random.random() > cfg.panel_border_prob:
        return

    draw = ImageDraw.Draw(page)
    w, h = page.size
    x0 = random.randint(cfg.margin, w // 2)
    y0 = random.randint(cfg.margin, h // 2)
    x1 = random.randint(x0 + 200, w - cfg.margin)
    y1 = random.randint(y0 + 200, h - cfg.margin)
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)


def add_page_number(page: Image.Image, cfg: PageConfig, font_paths: List[Path]) -> None:
    if random.random() > cfg.page_number_prob:
        return

    w, h = page.size
    font = load_font(font_paths, random.randint(16, 20))
    draw = ImageDraw.Draw(page)
    number = str(random.randint(1, 250))
    draw.text((w - cfg.margin - 40, h - cfg.margin + 10), number, font=font, fill=(0, 0, 0))


def add_rgroup_table(page: Image.Image, cfg: PageConfig, font_paths: List[Path]) -> None:
    if random.random() > cfg.rgroup_table_prob:
        return

    draw = ImageDraw.Draw(page)
    w, h = page.size
    x0 = random.randint(cfg.margin, w - cfg.margin - 240)
    y0 = random.randint(cfg.margin, h - cfg.margin - 160)
    cols = 3
    rows = 4
    cell_w = 80
    cell_h = 40

    for i in range(cols + 1):
        draw.line((x0 + i * cell_w, y0, x0 + i * cell_w, y0 + rows * cell_h), fill=(0, 0, 0), width=1)
    for j in range(rows + 1):
        draw.line((x0, y0 + j * cell_h, x0 + cols * cell_w, y0 + j * cell_h), fill=(0, 0, 0), width=1)

    font = load_font(font_paths, 14)
    for r in range(rows):
        for c in range(cols):
            txt = random.choice(["R1", "R2", "R3", "H", "Me", "Cl", "Br", "F", "OH", "OMe"])
            draw.text((x0 + c * cell_w + 6, y0 + r * cell_h + 6), txt, font=font, fill=(0, 0, 0))


def add_stray_text(page: Image.Image, cfg: PageConfig, font_paths: List[Path]) -> None:
    if random.random() > cfg.stray_text_prob:
        return

    draw = ImageDraw.Draw(page)
    w, h = page.size
    font = load_font(font_paths, random.randint(12, 16))
    fragments = [
        "J Med Chem",
        "DOI:10.1000/xyz",
        "Supplementary",
        "Table S1",
        "Scheme 3",
        "Rev. 2021",
    ]
    text = random.choice(fragments)
    x0 = random.randint(cfg.margin, w - cfg.margin - 160)
    y0 = random.randint(cfg.margin, h - cfg.margin - 40)
    draw.text((x0, y0), text, font=font, fill=(0, 0, 0))


def apply_noise(img: Image.Image, cfg: PageConfig) -> Image.Image:
    out = img

    if random.random() < cfg.brightness_prob:
        factor = random.uniform(0.85, 1.15)
        arr = np.array(out).astype(np.float32)
        arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

    if random.random() < cfg.blur_prob:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))

    if random.random() < cfg.noise_prob:
        arr = np.array(out).astype(np.int16)
        noise = np.random.normal(0, 6, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

    return out


PROSE_WORDS = [
    "the", "compound", "shows", "activity", "against", "cells", "in", "assay",
    "results", "indicate", "synthesis", "using", "method", "yield", "purity",
    "observed", "significant", "increase", "decrease", "binding", "affinity",
    "structure", "analysis", "reported", "table", "figure", "reaction",
]

CAPTION_TEMPLATES = [
    "Synthesis of target compounds.",
    "Representative structures from screening.",
    "Overview of reaction scheme.",
    "Chemical structures and labels.",
]


def make_page(
    smiles_pool: List[str],
    cfg: PageConfig,
    font_paths: List[Path],
) -> Tuple[Image.Image, List[Tuple[int, int, int, int]]]:
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), (255, 255, 255))

    existing_boxes: List[Tuple[int, int, int, int]] = []
    struct_boxes: List[Tuple[int, int, int, int]] = []

    n_structures = random.randint(cfg.min_structures, cfg.max_structures)
    random.shuffle(smiles_pool)

    for smi in smiles_pool[: n_structures * 2]:
        size = random.randint(*cfg.struct_size_range)
        struct_img = render_structure(smi, size, cfg)
        if struct_img is None:
            continue

        box = place_structure(page, struct_img, cfg, existing_boxes)
        if box is None:
            continue

        struct_boxes.append(box)
        existing_boxes.append(box)
        label_box = add_label_near_structure(page, box, cfg, font_paths)
        existing_boxes.append(label_box)

        if len(struct_boxes) >= n_structures:
            break

    add_prose_block(page, cfg, font_paths)
    add_caption(page, cfg, font_paths)
    add_arrow(page, cfg)
    add_panel_border(page, cfg)
    add_page_number(page, cfg, font_paths)
    add_rgroup_table(page, cfg, font_paths)
    add_stray_text(page, cfg, font_paths)

    return page, struct_boxes


def yolo_label(box: Tuple[int, int, int, int], w: int, h: int) -> str:
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def find_fonts(fonts_dir: Optional[Path]) -> List[Path]:
    if not fonts_dir or not fonts_dir.exists():
        return []
    font_paths = []
    for ext in ("*.ttf", "*.otf"):
        font_paths.extend(fonts_dir.rglob(ext))
    return font_paths


def save_sample(
    page: Image.Image,
    struct_boxes: List[Tuple[int, int, int, int]],
    out_img: Path,
    out_lbl: Path,
    fmt: str,
    cfg: PageConfig,
) -> None:
    page = apply_noise(page, cfg)

    if fmt.lower() == "jpg":
        page.save(out_img, format="JPEG", quality=random.randint(60, 90))
    else:
        page.save(out_img, format="PNG")

    labels = [yolo_label(box, cfg.page_w, cfg.page_h) for box in struct_boxes]
    out_lbl.write_text("\n".join(labels))


def generate_dataset(
    smiles_csv: Path,
    out_dir: Path,
    num_train: int,
    num_val: int,
    seed: int,
    fmt: str,
    fonts_dir: Optional[Path],
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    cfg = PageConfig()
    smiles_pool = load_smiles(smiles_csv)
    font_paths = find_fonts(fonts_dir)

    train_img_dir = out_dir / "train" / "images"
    train_lbl_dir = out_dir / "train" / "labels"
    val_img_dir = out_dir / "val" / "images"
    val_lbl_dir = out_dir / "val" / "labels"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    def run_split(count: int, img_dir: Path, lbl_dir: Path, split: str) -> None:
        for i in tqdm(range(count), desc=f"Generating {split}"):
            page, boxes = make_page(smiles_pool, cfg, font_paths)
            img_path = img_dir / f"{split}_{i:06d}.{fmt}"
            lbl_path = lbl_dir / f"{split}_{i:06d}.txt"
            save_sample(page, boxes, img_path, lbl_path, fmt, cfg)

    run_split(num_train, train_img_dir, train_lbl_dir, "train")
    run_split(num_val, val_img_dir, val_lbl_dir, "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for structure detection")
    parser.add_argument(
        "--smiles",
        type=Path,
        default=Path("data/smiles/chembl_smiles.csv"),
        help="Path to SMILES CSV",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/generated"),
        help="Output dataset directory",
    )
    parser.add_argument("--num-train", type=int, default=2000, help="Number of training pages")
    parser.add_argument("--num-val", type=int, default=200, help="Number of validation pages")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Image format",
    )
    parser.add_argument(
        "--fonts-dir",
        type=Path,
        default=None,
        help="Optional directory of .ttf/.otf fonts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.smiles.exists():
        print(f"SMILES CSV not found: {args.smiles}")
        return 1

    generate_dataset(
        smiles_csv=args.smiles,
        out_dir=args.out,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
        fmt=args.format,
        fonts_dir=args.fonts_dir,
    )

    print(f"\nDone. Dataset saved under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
