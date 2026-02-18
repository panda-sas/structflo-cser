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
    label_offset_range: Tuple[int, int] = (10, 20)
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

    # Use substantial padding to prevent clipping when rotated
    padding = max(20, int(max(text_w, text_h) * 0.3))
    padded_w = text_w + 2 * padding
    padded_h = text_h + 2 * padding
    
    text_img = Image.new("RGBA", (padded_w, padded_h), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((padding, padding), text, font=font, fill=fill)

    rotated = text_img.rotate(angle, expand=True)
    
    # Position the text so its center aligns with the requested position,
    # but stay within page bounds
    cx, cy = position
    page_w, page_h = base.size
    
    # Try to center the rotated text on the requested position
    x0 = cx - rotated.size[0] // 2
    y0 = cy - rotated.size[1] // 2
    
    # Clamp to page bounds
    x0 = max(0, min(x0, page_w - rotated.size[0]))
    y0 = max(0, min(y0, page_h - rotated.size[1]))
    
    base.paste(rotated, (x0, y0), rotated)

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

    return draw_rotated_text(page, label, (pos_x, pos_y), font, angle)


def try_place_box(
    w: int,
    h: int,
    box_w: int,
    box_h: int,
    margin: int,
    existing: List[Tuple[int, int, int, int]],
    max_tries: int = 50,
    padding: int = 8,
) -> Optional[Tuple[int, int, int, int]]:
    """Try to find a non-overlapping position for a box of given size."""
    for _ in range(max_tries):
        x0 = random.randint(margin, max(margin, w - margin - box_w))
        y0 = random.randint(margin, max(margin, h - margin - box_h))
        box = (x0, y0, x0 + box_w, y0 + box_h)
        padded = (x0 - padding, y0 - padding, x0 + box_w + padding, y0 + box_h + padding)
        if not any(boxes_intersect(padded, b) for b in existing):
            return box
    return None


def add_prose_block(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.prose_block_prob:
        return

    w, h = page.size
    font_size = random.randint(14, 18)
    font = load_font(font_paths, font_size)
    line_h = font_size + 4
    n_lines = random.randint(4, 12)
    block_w = random.randint(w // 4, w // 2)
    block_h = n_lines * line_h

    box = try_place_box(w, h, block_w, block_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)

    draw = ImageDraw.Draw(page)
    for i in range(n_lines):
        words = [random.choice(PROSE_WORDS) for _ in range(random.randint(6, 12))]
        draw.text((x0, y0 + i * line_h), " ".join(words), font=font, fill=(0, 0, 0))


def add_caption(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.caption_prob:
        return

    w, h = page.size
    font = load_font(font_paths, random.randint(16, 20))
    draw = ImageDraw.Draw(page)
    cap = f"Figure {random.randint(1, 12)}. {random.choice(CAPTION_TEMPLATES)}"

    text_bbox = draw.textbbox((0, 0), cap, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1] + 4

    box = try_place_box(w, h, text_w, text_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)
    draw.text((x0, y0), cap, font=font, fill=(0, 0, 0))


def add_arrow(
    page: Image.Image,
    cfg: PageConfig,
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.arrow_prob:
        return

    w, h = page.size
    length = random.randint(60, 140)
    arrow_h = 16

    box = try_place_box(w, h, length, arrow_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)

    draw = ImageDraw.Draw(page)
    mid_y = y0 + arrow_h // 2
    x1 = x0 + length
    draw.line((x0, mid_y, x1, mid_y), fill=(0, 0, 0), width=2)
    draw.line((x1, mid_y, x1 - 10, mid_y - 6), fill=(0, 0, 0), width=2)
    draw.line((x1, mid_y, x1 - 10, mid_y + 6), fill=(0, 0, 0), width=2)


def add_panel_border(
    page: Image.Image,
    cfg: PageConfig,
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.panel_border_prob:
        return

    draw = ImageDraw.Draw(page)
    w, h = page.size
    # Panel borders are large — just draw them (they frame content, overlap is natural)
    x0 = random.randint(cfg.margin, w // 2)
    y0 = random.randint(cfg.margin, h // 2)
    x1 = random.randint(x0 + 200, w - cfg.margin)
    y1 = random.randint(y0 + 200, h - cfg.margin)
    draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)


def add_page_number(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.page_number_prob:
        return

    w, h = page.size
    font = load_font(font_paths, random.randint(16, 20))
    draw = ImageDraw.Draw(page)
    number = str(random.randint(1, 250))
    # Page numbers go in corners — low overlap risk, just place them
    px = w - cfg.margin - 40
    py = h - cfg.margin + 10
    draw.text((px, py), number, font=font, fill=(0, 0, 0))


def add_rgroup_table(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.rgroup_table_prob:
        return

    w, h = page.size
    cols = random.randint(2, 4)
    rows = random.randint(3, 6)
    cell_w = random.randint(60, 100)
    cell_h = random.randint(30, 50)
    table_w = cols * cell_w
    table_h = rows * cell_h

    box = try_place_box(w, h, table_w, table_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)

    draw = ImageDraw.Draw(page)
    for i in range(cols + 1):
        draw.line((x0 + i * cell_w, y0, x0 + i * cell_w, y0 + rows * cell_h), fill=(0, 0, 0), width=1)
    for j in range(rows + 1):
        draw.line((x0, y0 + j * cell_h, x0 + cols * cell_w, y0 + j * cell_h), fill=(0, 0, 0), width=1)

    font = load_font(font_paths, 14)
    for r in range(rows):
        for c in range(cols):
            txt = random.choice(["R1", "R2", "R3", "H", "Me", "Cl", "Br", "F", "OH", "OMe", "NH2", "CF3"])
            draw.text((x0 + c * cell_w + 6, y0 + r * cell_h + 6), txt, font=font, fill=(0, 0, 0))


def add_stray_text(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    if random.random() > cfg.stray_text_prob:
        return

    w, h = page.size
    font_size = random.randint(12, 16)
    font = load_font(font_paths, font_size)
    text = random.choice(STRAY_FRAGMENTS)

    draw = ImageDraw.Draw(page)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0] + 4
    text_h = text_bbox[3] - text_bbox[1] + 4

    box = try_place_box(w, h, text_w, text_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)
    draw.text((x0, y0), text, font=font, fill=(0, 0, 0))


# ---------------------------------------------------------------------------
#  NEW: Random image / chart generators
# ---------------------------------------------------------------------------

def _gen_bar_chart(width: int, height: int) -> Image.Image:
    """Generate a simple random bar chart image."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n_bars = random.randint(3, 8)
    bar_w = max(4, (width - 40) // n_bars - 4)
    max_bar_h = height - 40
    colors = [
        (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
        (171, 71, 188), (0, 172, 193), (255, 112, 67), (117, 117, 117),
    ]
    x_start = 20
    for i in range(n_bars):
        bar_h = random.randint(max_bar_h // 5, max_bar_h)
        color = random.choice(colors)
        bx = x_start + i * (bar_w + 4)
        by = height - 20 - bar_h
        draw.rectangle([bx, by, bx + bar_w, height - 20], fill=color, outline=(0, 0, 0))
    # Axes
    draw.line((18, 15, 18, height - 18), fill=(0, 0, 0), width=2)
    draw.line((18, height - 18, width - 5, height - 18), fill=(0, 0, 0), width=2)
    return img


def _gen_scatter_plot(width: int, height: int) -> Image.Image:
    """Generate a simple random scatter plot image."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n_pts = random.randint(15, 60)
    color = random.choice([
        (66, 133, 244), (219, 68, 55), (15, 157, 88), (171, 71, 188),
    ])
    for _ in range(n_pts):
        cx = random.randint(25, width - 15)
        cy = random.randint(15, height - 25)
        r = random.randint(2, 5)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    # Axes
    draw.line((20, 10, 20, height - 18), fill=(0, 0, 0), width=2)
    draw.line((20, height - 18, width - 5, height - 18), fill=(0, 0, 0), width=2)
    return img


def _gen_line_plot(width: int, height: int) -> Image.Image:
    """Generate a simple random line plot image."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n_lines = random.randint(1, 3)
    colors = [(66, 133, 244), (219, 68, 55), (15, 157, 88)]
    for li in range(n_lines):
        n_pts = random.randint(5, 15)
        pts = []
        for i in range(n_pts):
            px = 25 + int(i * (width - 40) / max(1, n_pts - 1))
            py = random.randint(15, height - 25)
            pts.append((px, py))
        color = colors[li % len(colors)]
        for j in range(len(pts) - 1):
            draw.line([pts[j], pts[j + 1]], fill=color, width=2)
    draw.line((20, 10, 20, height - 18), fill=(0, 0, 0), width=2)
    draw.line((20, height - 18, width - 5, height - 18), fill=(0, 0, 0), width=2)
    return img


def _gen_pie_chart(size: int) -> Image.Image:
    """Generate a simple random pie chart image."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n_slices = random.randint(3, 7)
    values = [random.random() for _ in range(n_slices)]
    total = sum(values)
    colors = [
        (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
        (171, 71, 188), (0, 172, 193), (255, 112, 67),
    ]
    margin = 10
    bbox = [margin, margin, size - margin, size - margin]
    start = 0
    for i, v in enumerate(values):
        extent = (v / total) * 360
        draw.pieslice(bbox, start, start + extent, fill=colors[i % len(colors)], outline=(0, 0, 0))
        start += extent
    return img


def _gen_geometric_shapes(width: int, height: int) -> Image.Image:
    """Generate an image with random geometric shapes (simulates diagrams/logos)."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    n_shapes = random.randint(3, 10)
    colors = [
        (66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
        (171, 71, 188), (0, 172, 193), (200, 200, 200), (100, 100, 100),
    ]
    for _ in range(n_shapes):
        shape = random.choice(["rect", "ellipse", "line", "triangle"])
        color = random.choice(colors)
        if shape == "rect":
            rx0 = random.randint(0, width - 20)
            ry0 = random.randint(0, height - 20)
            rx1 = rx0 + random.randint(10, min(80, width - rx0))
            ry1 = ry0 + random.randint(10, min(80, height - ry0))
            if random.random() < 0.5:
                draw.rectangle([rx0, ry0, rx1, ry1], fill=color)
            else:
                draw.rectangle([rx0, ry0, rx1, ry1], outline=color, width=2)
        elif shape == "ellipse":
            cx = random.randint(10, width - 10)
            cy = random.randint(10, height - 10)
            rx = random.randint(5, 40)
            ry = random.randint(5, 40)
            draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color)
        elif shape == "line":
            lx0 = random.randint(0, width)
            ly0 = random.randint(0, height)
            lx1 = random.randint(0, width)
            ly1 = random.randint(0, height)
            draw.line([lx0, ly0, lx1, ly1], fill=color, width=random.randint(1, 3))
        elif shape == "triangle":
            pts = [(random.randint(0, width), random.randint(0, height)) for _ in range(3)]
            draw.polygon(pts, fill=color)
    return img


def _gen_noise_patch(width: int, height: int) -> Image.Image:
    """Generate a patch of random noise (simulates a scan artifact or photo region)."""
    arr = np.random.randint(180, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _gen_gradient_block(width: int, height: int) -> Image.Image:
    """Generate a gradient block (simulates a shaded region or header bar)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    c1 = np.array([random.randint(180, 240)] * 3, dtype=np.float64)
    c2 = np.array([random.randint(220, 255)] * 3, dtype=np.float64)
    for y in range(height):
        t = y / max(1, height - 1)
        row_color = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        arr[y, :] = row_color
    return Image.fromarray(arr)


RANDOM_IMAGE_GENERATORS = [
    lambda: _gen_bar_chart(random.randint(180, 400), random.randint(140, 300)),
    lambda: _gen_scatter_plot(random.randint(180, 400), random.randint(140, 300)),
    lambda: _gen_line_plot(random.randint(180, 400), random.randint(140, 300)),
    lambda: _gen_pie_chart(random.randint(140, 280)),
    lambda: _gen_geometric_shapes(random.randint(120, 350), random.randint(120, 350)),
    lambda: _gen_noise_patch(random.randint(100, 300), random.randint(80, 200)),
    lambda: _gen_gradient_block(random.randint(200, 500), random.randint(40, 120)),
]


def add_random_image(
    page: Image.Image,
    cfg: PageConfig,
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Paste a randomly generated chart / shape / image onto the page."""
    gen = random.choice(RANDOM_IMAGE_GENERATORS)
    rand_img = gen()
    iw, ih = rand_img.size
    w, h = page.size

    box = try_place_box(w, h, iw, ih, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)
    page.paste(rand_img, (x0, y0))


# ---------------------------------------------------------------------------
#  NEW: Richer random text generators
# ---------------------------------------------------------------------------

HEADER_TEMPLATES = [
    "Abstract", "Introduction", "Methods", "Results", "Discussion",
    "Experimental Section", "Conclusions", "References", "Supplementary Information",
    "Materials and Methods", "Acknowledgments", "Author Contributions",
]

FOOTNOTE_TEMPLATES = [
    "* Corresponding author. Email: author@university.edu",
    "† These authors contributed equally.",
    "‡ Current address: Department of Chemistry, MIT.",
    "§ Electronic supplementary information (ESI) available.",
    "a) Reagents and conditions: see text.",
    "b) Isolated yield after column chromatography.",
]

EQUATION_FRAGMENTS = [
    "IC50 = 3.2 ± 0.5 nM",
    "Ki = 12.4 ± 1.1 µM",
    "EC50 = 0.8 µM (n=3)",
    "ΔG = -8.3 kcal/mol",
    "logP = 2.14",
    "MW = 423.5 g/mol",
    "t1/2 = 4.2 h",
    "Kd = 45 nM",
    "%inh = 87 ± 3%",
    "AUC = 1240 ng·h/mL",
]

JOURNAL_HEADERS = [
    "Journal of Medicinal Chemistry",
    "Bioorganic & Medicinal Chemistry Letters",
    "European Journal of Medicinal Chemistry",
    "ACS Chemical Biology",
    "Nature Chemical Biology",
    "Chemical Communications",
    "Angewandte Chemie Int. Ed.",
    "Organic Letters",
    "Tetrahedron Letters",
]


def add_section_header(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Add a bold section header like 'Abstract', 'Methods', etc."""
    w, h = page.size
    text = random.choice(HEADER_TEMPLATES)
    font_size = random.randint(20, 30)
    font = load_font(font_paths, font_size)
    draw = ImageDraw.Draw(page)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0] + 4, tb[3] - tb[1] + 4

    box = try_place_box(w, h, tw, th, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)
    draw.text((x0, y0), text, font=font, fill=(0, 0, 0))


def add_footnote(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Add a footnote-style text near the bottom of the page."""
    w, h = page.size
    text = random.choice(FOOTNOTE_TEMPLATES)
    font_size = random.randint(10, 14)
    font = load_font(font_paths, font_size)
    draw = ImageDraw.Draw(page)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0] + 4, tb[3] - tb[1] + 4

    # Prefer bottom quarter of page
    for _ in range(30):
        x0 = random.randint(cfg.margin, max(cfg.margin, w - cfg.margin - tw))
        y0 = random.randint(max(cfg.margin, h * 3 // 4), max(cfg.margin, h - cfg.margin - th))
        candidate = (x0, y0, x0 + tw, y0 + th)
        padded = (x0 - 8, y0 - 8, x0 + tw + 8, y0 + th + 8)
        if not any(boxes_intersect(padded, b) for b in existing):
            existing.append(candidate)
            draw.text((x0, y0), text, font=font, fill=(80, 80, 80))
            return


def add_equation_fragment(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Add a small scientific equation / measurement fragment."""
    w, h = page.size
    text = random.choice(EQUATION_FRAGMENTS)
    font_size = random.randint(14, 20)
    font = load_font(font_paths, font_size)
    draw = ImageDraw.Draw(page)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0] + 4, tb[3] - tb[1] + 4

    box = try_place_box(w, h, tw, th, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)
    draw.text((x0, y0), text, font=font, fill=(0, 0, 0))


def add_journal_header(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Add a journal-name style header near the top of the page."""
    w, h = page.size
    text = random.choice(JOURNAL_HEADERS)
    font_size = random.randint(14, 20)
    font = load_font(font_paths, font_size)
    draw = ImageDraw.Draw(page)
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0] + 4, tb[3] - tb[1] + 4

    # Prefer top region of page
    for _ in range(30):
        x0 = random.randint(cfg.margin, max(cfg.margin, w - cfg.margin - tw))
        y0 = random.randint(10, min(cfg.margin + 60, h - th))
        candidate = (x0, y0, x0 + tw, y0 + th)
        padded = (x0 - 8, y0 - 8, x0 + tw + 8, y0 + th + 8)
        if not any(boxes_intersect(padded, b) for b in existing):
            existing.append(candidate)
            draw.text((x0, y0), text, font=font, fill=(60, 60, 60))
            return


def add_multiline_text_block(
    page: Image.Image,
    cfg: PageConfig,
    font_paths: List[Path],
    existing: List[Tuple[int, int, int, int]],
) -> None:
    """Add a multi-line block of random scientific-ish text."""
    w, h = page.size
    font_size = random.randint(12, 16)
    font = load_font(font_paths, font_size)
    line_h = font_size + 3
    n_lines = random.randint(3, 8)
    block_w = random.randint(w // 5, w // 3)
    block_h = n_lines * line_h

    box = try_place_box(w, h, block_w, block_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)

    draw = ImageDraw.Draw(page)
    extended = PROSE_WORDS + [
        "inhibitor", "selectivity", "potent", "analog", "derivative",
        "molecular", "weight", "peak", "area", "concentration",
        "dose", "response", "curve", "receptor", "antagonist",
        "pharmacokinetic", "bioavailability", "clearance", "metabolite",
    ]
    for i in range(n_lines):
        words = [random.choice(extended) for _ in range(random.randint(4, 10))]
        draw.text((x0, y0 + i * line_h), " ".join(words), font=font, fill=(0, 0, 0))


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
    "series", "scaffold", "moiety", "substituent", "modification", "optimization",
    "efficacy", "toxicity", "vitro", "vivo", "plasma", "solubility", "target",
    "potency", "screening", "library", "fragment", "lead", "candidate",
]

CAPTION_TEMPLATES = [
    "Synthesis of target compounds.",
    "Representative structures from screening.",
    "Overview of reaction scheme.",
    "Chemical structures and labels.",
    "SAR summary for the lead series.",
    "Dose-response curves for selected compounds.",
    "X-ray crystal structure of the protein-ligand complex.",
    "Proposed mechanism of action.",
    "Metabolic stability across species.",
]

STRAY_FRAGMENTS = [
    "J Med Chem", "DOI:10.1000/xyz", "Supplementary", "Table S1",
    "Scheme 3", "Rev. 2021", "pKa = 7.4", "cLogP 3.2",
    "HPLC purity >95%", "mp 142-145 °C", "[α]D = -23.5",
    "HRMS (ESI)", "1H NMR (400 MHz)", "13C NMR (101 MHz)",
    "Received: Jan 2025", "Accepted: Mar 2025", "Published online",
    "Supporting Information", "Author Manuscript", "CONFIDENTIAL",
    "Patent WO2025/123456", "© 2025 ACS", "All rights reserved.",
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

    # -- Distractor: text elements (overlap-aware) --
    add_prose_block(page, cfg, font_paths, existing_boxes)
    add_caption(page, cfg, font_paths, existing_boxes)
    add_page_number(page, cfg, font_paths, existing_boxes)
    add_rgroup_table(page, cfg, font_paths, existing_boxes)
    add_stray_text(page, cfg, font_paths, existing_boxes)
    add_arrow(page, cfg, existing_boxes)
    add_panel_border(page, cfg, existing_boxes)

    # -- NEW: Additional random text blocks --
    if random.random() < 0.5:
        add_section_header(page, cfg, font_paths, existing_boxes)
    if random.random() < 0.4:
        add_footnote(page, cfg, font_paths, existing_boxes)
    if random.random() < 0.35:
        add_equation_fragment(page, cfg, font_paths, existing_boxes)
    if random.random() < 0.3:
        add_journal_header(page, cfg, font_paths, existing_boxes)
    # Extra multiline blocks (1–2)
    for _ in range(random.randint(0, 2)):
        if random.random() < 0.45:
            add_multiline_text_block(page, cfg, font_paths, existing_boxes)
    # Extra stray text snippets (0–3)
    for _ in range(random.randint(0, 3)):
        if random.random() < 0.5:
            add_stray_text(page, cfg, font_paths, existing_boxes)

    # -- NEW: Random images / charts (1–3) --
    n_images = random.randint(0, 3)
    for _ in range(n_images):
        if random.random() < 0.5:
            add_random_image(page, cfg, existing_boxes)

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
