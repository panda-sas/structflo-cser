#!/usr/bin/env python3
"""
Synthetic page generator with distractors.

Generates document-like pages containing:
  - Chemical structures with nearby label IDs, annotated together as a single
    compound-panel bounding box (class 0 for YOLO — union of struct + label)
  - Distractor elements: prose, captions, arrows, panel borders, page numbers
    (NOT annotated — implicit negatives for YOLO)

Output: YOLO-format dataset (images/ + labels/ with union struct+label bboxes).
         Ground-truth JSON (ground_truth/) with split struct/label boxes + text.
"""

import argparse
import csv
import json
import math
import os
import random
import string
import textwrap
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
    max_structures: int = 10
    two_column_prob: float = 0.25
    grid_layout_prob: float = 0.20   # 20% chance of uniform grid layout
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


def make_page_config(dpi: int = 300) -> PageConfig:
    """Return a PageConfig with all pixel dimensions scaled for *dpi*.

    The canonical reference is 300 DPI (A4 = 2480×3508 px).
    Pass dpi=144 to get a ~1191×1684 px page matching real-world scans.
    """
    s = dpi / 300.0
    return PageConfig(
        page_w=int(2480 * s),
        page_h=int(3508 * s),
        margin=int(180 * s),
        struct_size_range=(max(60, int(280 * s)), max(100, int(550 * s))),
        label_font_range=(max(8, int(12 * s)), max(12, int(36 * s))),
        label_offset_range=(max(3, int(10 * s)), max(6, int(20 * s))),
    )


def make_page_config_slide(dpi: int = 96) -> PageConfig:
    """Return a PageConfig for landscape 16:9 slide format (PowerPoint/Keynote PDFs).

    Reference: 13.33" × 7.5" at 96 DPI → 1280×720 px.
    Structures are somewhat smaller relative to portrait A4 since slides
    pack fewer compounds — more whitespace per structure.
    """
    s = dpi / 96.0
    return PageConfig(
        page_w=int(1280 * s),
        page_h=int(720 * s),
        margin=int(50 * s),
        struct_size_range=(max(60, int(180 * s)), max(100, int(340 * s))),
        label_font_range=(max(8, int(10 * s)), max(12, int(26 * s))),
        label_offset_range=(max(3, int(8 * s)), max(6, int(14 * s))),
        min_structures=1,
        max_structures=6,        # slides pack fewer compounds
        two_column_prob=0.20,
        grid_layout_prob=0.35,   # grids are common in slide figures
    )


def _rand_prefix(min_len: int = 3, max_len: int = 6) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=random.randint(min_len, max_len)))


LABEL_STYLES = {
    # CHEMBL2000, ZINC123456 — well-known DB prefix + 4-7 digits
    "chembl_like": lambda: (
        random.choice(["CHEMBL", "PUBCHEM", "ZINC", "MCULE", "ENAMINE"])
        + str(random.randint(100, 9_999_999))
    ),
    # SACC-33000, MERK-5512 — 3-5 uppercase letters + dash + digits
    "dash_long": lambda: (
        _rand_prefix(3, 5) + "-" + str(random.randint(100, 999_999))
    ),
    # LGNIA55, BXTR2204 — 4-6 uppercase letters directly followed by digits (no dash)
    "prefix_nodash": lambda: (
        _rand_prefix(4, 6) + str(random.randint(10, 99999))
    ),
    # MERK-22.4.5.6 — prefix + dash + dotted hierarchical number
    "dotted_hierarchy": lambda: (
        _rand_prefix(3, 5) + "-"
        + ".".join(str(random.randint(1, 99)) for _ in range(random.randint(2, 4)))
    ),
    # CAS: 50-78-2, 1234-56-7
    "cas_like": lambda: (
        f"{random.randint(50, 99999)}-{random.randint(10, 99)}-{random.randint(0, 9)}"
    ),
    # Internal catalog: CPD-00123, HIT-04567
    "catalog": lambda: (
        random.choice(["CPD", "MOL", "HIT", "REF", "STD", "LIB", "SCR", "CMP", "SYN"])
        + "-" + str(random.randint(1, 99999)).zfill(5)
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
    """Render a 2-D chemical structure from a SMILES string.

    Uses RDKit's Cairo drawer to produce a transparent-background RGBA image,
    then tight-crops around the drawn atoms/bonds so there's no excess whitespace.

    Returns None if the SMILES is invalid or produces no visible drawing.
    """
    # Parse the SMILES into an RDKit molecule object.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        # Generate 2-D coordinates for drawing.
        AllChem.Compute2DCoords(mol)
    except Exception:
        return None

    # Set up the Cairo-based drawer at the requested pixel size.
    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = random.uniform(*cfg.bond_width_range)    # vary line thickness
    opts.minFontSize = random.randint(*cfg.atom_font_range)       # vary atom-label size
    opts.maxFontSize = opts.minFontSize + 8
    opts.additionalAtomLabelPadding = random.uniform(0.05, 0.2)   # breathing room around labels
    opts.rotate = random.uniform(0, 360)                          # random orientation
    if random.random() < 0.3:
        opts.useBWAtomPalette()  # 30 % chance: all-black atoms (no colour)

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Read the rendered PNG bytes into a Pillow RGBA image.
    img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
    arr = np.array(img)

    # Build a mask of "meaningful" pixels: non-transparent AND not pure white.
    # This lets us tight-crop the structure, discarding the white canvas.
    mask = (arr[:, :, 3] > 0) & (
        (arr[:, :, 0] < 250) | (arr[:, :, 1] < 250) | (arr[:, :, 2] < 250)
    )
    if not mask.any():
        return None

    # Find the bounding rectangle of the drawn content and crop.
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
    return cropped


def load_font(font_paths: List[Path], size: int, prefer_bold: bool = False) -> ImageFont.ImageFont:
    """Pick a random font from *font_paths* at the given pixel *size*.

    If *prefer_bold* is True, only bold-named fonts are considered (if any).
    Falls back to DejaVuSans or Pillow's built-in default on failure.
    """
    paths = list(font_paths)
    if prefer_bold:
        bold_paths = [p for p in paths if "bold" in p.name.lower()]
        if bold_paths:
            paths = bold_paths
    random.shuffle(paths)  # randomise so each call picks a different typeface
    for path in paths:
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue
    # Fallback chain: system DejaVuSans -> Pillow bitmap default.
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
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> Tuple[int, int, int, int]:
    """Render *text* onto *base* at the given centre-point, optionally rotated.

    Parameters
    ----------
    base : Image
        The page image to draw on (modified in-place).
    text : str
        The label string to render.
    position : (cx, cy)
        The **centre** of the desired label position on the page.
    font : ImageFont
        Pillow font object to use for rendering.
    angle : float
        Counter-clockwise rotation in degrees (0 = horizontal).
    fill : (R, G, B)
        Text colour.
    bg_color : (R, G, B) or None
        If given, a filled rectangle of this colour is drawn behind the text
        (e.g. dark background with light text).

    Returns
    -------
    (x0, y0, x1, y1) — tight bounding box of the rendered label on *base*,
    padded by `annotation_margin` pixels on each side.
    """
    draw = ImageDraw.Draw(base)

    # --- Step 1: Measure the text ------------------------------------------
    # textbbox returns (left, top, right, bottom) for the given string.
    # Note: left/top may be non-zero because of font metrics (ascent offset),
    # so we subtract them to get the true glyph width/height.
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]   # pixel width of the text glyphs
    text_h = text_bbox[3] - text_bbox[1]   # pixel height of the text glyphs

    # --- Step 2: Determine canvas padding -----------------------------------
    # We draw the text on a temporary transparent canvas, then rotate it.
    #   • annotation_margin : extra pixels kept around the text in the final
    #     returned bounding box (so the YOLO bbox is not skin-tight).
    #   • canvas_padding : extra transparent border around the text on the
    #     temporary canvas.  Must be >= annotation_margin so the tight-bbox
    #     scan (step 6) never bumps into the canvas edge.
    #     For rotated text the diagonal is larger, so we add more padding.
    annotation_margin = 7
    if abs(angle) < 0.5:
        # No rotation: a small border is enough.
        canvas_padding = annotation_margin + 2
    else:
        # Rotation: corners sweep outward by (diag − short_side)/2.
        diag = int(np.hypot(text_w, text_h))
        canvas_padding = (diag - min(text_w, text_h)) // 2 + annotation_margin + 4

    padded_w = text_w + 2 * canvas_padding
    padded_h = text_h + 2 * canvas_padding

    # --- Step 3: Create a temporary RGBA canvas and draw text ---------------
    # The canvas is fully transparent; only the text (and optional bg rect)
    # will have non-zero alpha — this lets us composite and measure later.
    text_img = Image.new("RGBA", (padded_w, padded_h), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)

    # The text draw origin inside the canvas, compensating for font offsets.
    # Pillow's textbbox((0,0), ...) returns (left, top, right, bottom) where
    # left/top are the font-metric offsets from the anchor.  By subtracting
    # them we ensure the actual glyph pixels land at (canvas_padding, canvas_padding).
    tx = canvas_padding - text_bbox[0]
    ty = canvas_padding - text_bbox[1]

    # Glyph pixel extent on the canvas (independent of font-metric offsets).
    glyph_x0 = canvas_padding
    glyph_y0 = canvas_padding
    glyph_x1 = canvas_padding + text_w
    glyph_y1 = canvas_padding + text_h

    if bg_color is not None:
        # Draw a filled background rectangle centred on the actual glyph area.
        # IMPORTANT: we use glyph coordinates, NOT (tx, ty), because those
        # include the font-metric offsets.  Using (tx ± bg_pad) would shift
        # the rect left/up, clipping white text on the right/bottom edges
        # (invisible against a white page).
        bg_pad = 4
        text_draw.rectangle(
            [glyph_x0 - bg_pad, glyph_y0 - bg_pad,
             glyph_x1 + bg_pad, glyph_y1 + bg_pad],
            fill=bg_color + (255,),
        )

    # Draw the actual text on top of the (optional) background rectangle.
    text_draw.text((tx, ty), text, font=font, fill=fill + (255,))

    # --- Step 4: Rotate the temporary canvas --------------------------------
    # expand=True grows the output image so nothing is cropped during rotation.
    rotated = text_img.rotate(angle, expand=True)

    # --- Step 5: Paste the rotated label centred on `position` --------------
    # `position` is the desired centre-point on the page, so we shift left/up
    # by half the rotated image dimensions, then clamp to stay on the page.
    cx, cy = position
    page_w, page_h = base.size
    x0 = cx - rotated.size[0] // 2
    y0 = cy - rotated.size[1] // 2
    x0 = max(0, min(x0, page_w - rotated.size[0]))  # clamp horizontally
    y0 = max(0, min(y0, page_h - rotated.size[1]))  # clamp vertically

    # Composite onto the page using the alpha channel as mask.
    base.paste(rotated, (x0, y0), rotated)

    # --- Step 6: Compute tight bounding box on the page ---------------------
    # Scan the alpha channel of the (possibly rotated) canvas to find the
    # non-transparent extent, then add annotation_margin on each side.
    # Because canvas_padding >= annotation_margin, subtracting annotation_margin
    # from the min coordinates will never go below 0 (so the bbox is valid).
    alpha = np.array(rotated)[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(ys) == 0:
        # Fallback: the whole rotated canvas region (shouldn't normally happen).
        return (x0, y0, x0 + rotated.size[0], y0 + rotated.size[1])
    bx0 = max(0, xs.min() - annotation_margin)
    by0 = max(0, ys.min() - annotation_margin)
    bx1 = min(rotated.size[0], xs.max() + 1 + annotation_margin)
    by1 = min(rotated.size[1], ys.max() + 1 + annotation_margin)
    return (int(x0 + bx0), int(y0 + by0), int(x0 + bx1), int(y0 + by1))


def clamp_box(box: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    """Clamp a bounding box so it stays within the page (0,0)-(w,h)."""
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(1, min(x1, w))
    y1 = max(1, min(y1, h))
    return x0, y0, x1, y1


def boxes_intersect(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    """Return True if axis-aligned boxes *a* and *b* overlap at all."""
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def place_structure(
    page: Image.Image,
    struct_img: Image.Image,
    cfg: PageConfig,
    existing_boxes: List[Tuple[int, int, int, int]],
    max_tries: int = 80,
    x_range: Optional[Tuple[int, int]] = None,
    y_range: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Try to paste *struct_img* onto *page* without overlapping existing boxes.

    Randomly samples positions within the allowed region (x_range/y_range or
    page margins) up to *max_tries* times.  Returns the (x0,y0,x1,y1) box on
    success, or None if no free position was found.
    """
    w, h = page.size
    sw, sh = struct_img.size  # structure image dimensions

    # Determine the allowed placement rectangle.
    x_lo = x_range[0] if x_range else cfg.margin
    x_hi = (x_range[1] - sw) if x_range else (w - cfg.margin - sw)
    y_lo = y_range[0] if y_range else cfg.margin
    y_hi = (y_range[1] - sh) if y_range else (h - cfg.margin - sh)

    if x_lo >= x_hi or y_lo >= y_hi:
        return None  # structure doesn't fit in the allowed region

    for _ in range(max_tries):
        x = random.randint(x_lo, x_hi)
        y = random.randint(y_lo, y_hi)
        box = (x, y, x + sw, y + sh)
        # Pad the candidate box by 6 px on each side to keep a gap.
        padded = (x - 6, y - 6, x + sw + 6, y + sh + 6)

        if any(boxes_intersect(padded, b) for b in existing_boxes):
            continue  # overlaps — try again

        # Found a free spot: composite the structure onto the page.
        page.paste(struct_img, (x, y), struct_img)
        return box
    return None  # exhausted attempts


def add_label_near_structure(
    page: Image.Image,
    struct_box: Tuple[int, int, int, int],
    cfg: PageConfig,
    font_paths: List[Path],
) -> Tuple[Tuple[int, int, int, int], str]:
    """Place a randomly-generated compound label next to *struct_box*.

    The label is positioned on a randomly chosen side of the structure
    (below / above / left / right) and may optionally be rotated and/or
    drawn with a dark background behind light text.

    Returns
    -------
    (label_bbox, label_text)
        label_bbox : (x0, y0, x1, y1) on the page after rendering.
        label_text : the string that was drawn.
    """
    w, h = page.size
    label = random_label()  # e.g. "CHEMBL12345" or "CPD-00042"

    # Pick a random font size within the configured range, optionally bold.
    font_size = random.randint(*cfg.label_font_range)
    use_bold = random.random() < 0.5
    font = load_font(font_paths, font_size, prefer_bold=use_bold)

    x0, y0, x1, y1 = struct_box  # bounding box of the chemical structure

    # Measure the text so we can compute placement coordinates.
    draw = ImageDraw.Draw(page)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]   # glyph width
    text_h = text_bbox[3] - text_bbox[1]   # glyph height

    # --- Choose which side of the structure the label goes on ---------------
    # Weighted random: below is most common (mimics real papers).
    placement = random.choices(
        ["below", "above", "left", "right"],
        weights=[50, 20, 15, 15],
    )[0]
    offset = random.randint(*cfg.label_offset_range)  # gap between struct and label

    # Compute the *centre* of the label for each placement.
    # draw_rotated_text() expects a centre point (cx, cy), NOT a top-left corner.
    struct_cx = x0 + (x1 - x0) // 2  # horizontal centre of the structure box
    struct_cy = y0 + (y1 - y0) // 2  # vertical centre of the structure box

    if placement == "below":
        # Horizontally centred under the structure, shifted down by offset.
        cx = struct_cx
        cy = y1 + offset + text_h // 2
        angle = 0.0
    elif placement == "above":
        # Horizontally centred above the structure, shifted up by offset.
        cx = struct_cx
        cy = y0 - offset - text_h // 2
        angle = 0.0
    elif placement == "left":
        # Vertically centred to the left of the structure.
        cx = x0 - offset - text_w // 2
        cy = struct_cy
        angle = 0.0
    else:  # right
        # Vertically centred to the right of the structure.
        cx = x1 + offset + text_w // 2
        cy = struct_cy
        angle = 0.0

    # --- Optional rotation --------------------------------------------------
    # Small chance of 90° rotation on side placements (mimics rotated axis labels).
    if placement in ("left", "right") and random.random() < cfg.label_90deg_prob:
        angle = 90.0
    elif random.random() < cfg.label_rotation_prob:
        # Slight tilt (e.g. −15° to +15°) to add variety.
        angle = random.uniform(*cfg.label_rotation_range)

    # --- Optional dark-background style (30 % of the time) ------------------
    # Mimics inverted / highlighted labels seen in some publications.
    if random.random() < 0.30:
        bg_color = random.choice([
            (0, 0, 0),        # black
            (30, 30, 30),     # dark grey
            (0, 0, 100),      # dark blue
            (80, 0, 0),       # dark red
            (0, 60, 0),       # dark green
            (60, 0, 80),      # dark purple
        ])
        fill_color = (255, 255, 255)  # white text on dark background
    else:
        bg_color = None
        fill_color = (0, 0, 0)        # default: black text, no background

    # Render the label onto the page and get its bounding box back.
    label_box = draw_rotated_text(page, label, (cx, cy), font, angle,
                                  fill=fill_color, bg_color=bg_color)
    return label_box, label


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
    n_lines = random.randint(8, 22)
    block_w = random.randint(w // 3, w * 3 // 4)
    block_h = n_lines * line_h

    box = try_place_box(w, h, block_w, block_h, cfg.margin, existing)
    if box is None:
        return
    x0, y0, _, _ = box
    existing.append(box)

    draw = ImageDraw.Draw(page)
    total_words = n_lines * random.randint(6, 10)
    words = [random.choice(PROSE_WORDS) for _ in range(total_words)]
    paragraph = " ".join(words).capitalize() + "."
    avg_char_w = font_size * 0.55
    chars_per_line = max(20, int(block_w / avg_char_w))
    lines = textwrap.wrap(paragraph, width=chars_per_line)[:n_lines]
    for i, line in enumerate(lines):
        draw.text((x0, y0 + i * line_h), line, font=font, fill=(0, 0, 0))


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
    # Mix standard R-group values with ID-like strings (hard negatives for compound_label)
    cell_choices = [
        "R1", "R2", "R3", "H", "Me", "Cl", "Br", "F", "OH", "OMe", "NH2", "CF3",
        "Et", "iPr", "nBu", "Ph", "Bn", "CN", "NO2", "SO2Me",
        "n/a", ">100", "<0.1", "3.2", "14.5",
    ]
    id_choices = [
        "CHEMBL4051", "ZINC00123", "CPD-00441", "MOL-98231",
        "HIT-00923", "STD-00001", "MCULE-001", "PUBCHEM44",
    ]
    for r in range(rows):
        for c in range(cols):
            # Header row or first column: sometimes use an ID as a row key
            if (r == 0 or c == 0) and random.random() < 0.25:
                txt = random.choice(id_choices)
            else:
                txt = random.choice(cell_choices)
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


def load_distractor_images(distractors_dir: Optional[Path]) -> List[Image.Image]:
    """Pre-load distractor images from disk (resized to manageable sizes)."""
    if distractors_dir is None or not distractors_dir.exists():
        return []
    imgs: List[Image.Image] = []
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(distractors_dir.glob(ext))
    for p in sorted(paths):
        try:
            img = Image.open(p).convert("RGB")
            # Don't resize here — we resize at paste time for variety
            imgs.append(img)
        except Exception:
            continue
    return imgs


def _pick_distractor_image(
    distractor_pool: List[Image.Image],
) -> Image.Image:
    """Pick a random real distractor image and resize it to a random distractor size."""
    img = random.choice(distractor_pool).copy()
    # Random target size typical for a document inset image
    target_w = random.randint(150, 500)
    target_h = random.randint(120, 400)
    img = img.resize((target_w, target_h), Image.LANCZOS)
    # Optionally convert to grayscale (documents are often B&W)
    if random.random() < 0.25:
        img = img.convert("L").convert("RGB")
    return img


def add_random_image(
    page: Image.Image,
    cfg: PageConfig,
    existing: List[Tuple[int, int, int, int]],
    distractor_pool: Optional[List[Image.Image]] = None,
) -> None:
    """Paste a distractor image onto the page.

    Uses real images from *distractor_pool* when available (85% of the time),
    falling back to synthetic chart/shape generators.
    """
    use_real = (
        distractor_pool
        and len(distractor_pool) > 0
        and random.random() < 0.85
    )
    if use_real:
        rand_img = _pick_distractor_image(distractor_pool)
    else:
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
    block_w = random.randint(w // 3, w * 2 // 3)
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
    total_words = n_lines * random.randint(4, 8)
    words = [random.choice(extended) for _ in range(total_words)]
    paragraph = " ".join(words).capitalize() + "."
    avg_char_w = font_size * 0.55
    chars_per_line = max(20, int(block_w / avg_char_w))
    lines = textwrap.wrap(paragraph, width=chars_per_line)[:n_lines]
    for i, line in enumerate(lines):
        draw.text((x0, y0 + i * line_h), line, font=font, fill=(0, 0, 0))


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
    # ── Hard negatives: ID-like strings in non-label context ──────────────────
    # These look like compound labels but appear as stray text, NOT annotated.
    "See CHEMBL4051 for details", "ZINC00123456 (inactive)",
    "Ref: PUBCHEM2341157", "cf. MCULE-1234567",
    "CPD-00441 showed no activity", "data from MOL-98231",
    "activity vs. ENAMINE-T001", "HIT-00923 excluded",
    "IC50 values: STD-00001–STD-00010",
    "Selected from ZINC library", "PUBCHEM CID: 44259",
]


def make_negative_page(
    cfg: PageConfig,
    font_paths: List[Path],
    distractor_pool: Optional[List[Image.Image]] = None,
) -> Tuple[Image.Image, List[dict]]:
    """Generate a page with no chemical structures — pure text/charts.

    Used as hard negatives so the model learns to output nothing when
    there are no structures present.
    """
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), (255, 255, 255))
    existing_boxes: List[Tuple[int, int, int, int]] = []

    for _ in range(random.randint(3, 6)):
        add_prose_block(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(3, 6)):
        add_multiline_text_block(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(1, 4)):
        add_caption(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(5, 10)):
        add_stray_text(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(2, 4)):
        add_equation_fragment(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(1, 3)):
        add_section_header(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(0, 2)):
        add_footnote(page, cfg, font_paths, existing_boxes)
    if random.random() < 0.8:
        add_journal_header(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(1, 3)):
        add_rgroup_table(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(0, 2)):
        add_arrow(page, cfg, existing_boxes)
    add_page_number(page, cfg, font_paths, existing_boxes)

    if distractor_pool:
        n_images = random.randint(2, 5)
    else:
        n_images = random.randint(1, 4)
    for _ in range(n_images):
        add_random_image(page, cfg, existing_boxes, distractor_pool)

    return page, []  # empty panels — no annotations


def make_page(
    smiles_pool: List[str],
    cfg: PageConfig,
    font_paths: List[Path],
    distractor_pool: Optional[List[Image.Image]] = None,
) -> Tuple[Image.Image, List[dict]]:
    """Generate one synthetic document page containing chemical structures.

    Steps:
      1. Decide layout mode (free-form / grid / two-column).
      2. For each structure: render it, place it on the page, add a nearby label.
      3. Scatter distractor elements (prose, captions, arrows, images, etc.).

    Returns (page_image, panels) where each panel dict has:
      struct_box, label_box, label_text, smiles.
    """
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), (255, 255, 255))

    existing_boxes: List[Tuple[int, int, int, int]] = []  # collision-avoidance registry
    panels: List[dict] = []  # one entry per placed structure+label pair

    n_structures = random.randint(cfg.min_structures, cfg.max_structures)
    random.shuffle(smiles_pool)

    # Choose layout mode
    layout_roll = random.random()
    use_grid = layout_roll < cfg.grid_layout_prob and n_structures >= 4
    two_column = (not use_grid) and layout_roll < cfg.grid_layout_prob + cfg.two_column_prob and n_structures >= 2

    if use_grid:
        # Uniform grid: pick cols × rows that fits n_structures
        cols = random.choice([2, 3, 4])
        rows = (n_structures + cols - 1) // cols
        usable_w = cfg.page_w - 2 * cfg.margin
        usable_h = cfg.page_h - 2 * cfg.margin
        cell_w = usable_w // cols
        cell_h = usable_h // rows
        grid_positions = [
            (cfg.margin + c * cell_w, cfg.margin + r * cell_h,
             cfg.margin + (c + 1) * cell_w, cfg.margin + (r + 1) * cell_h)
            for r in range(rows) for c in range(cols)
        ]
        col_x_ranges = None
        col_y_range = None
    elif two_column:
        col_gap = 60
        mid = cfg.page_w // 2
        col_x_ranges = [
            (cfg.margin, mid - col_gap // 2),
            (mid + col_gap // 2, cfg.page_w - cfg.margin),
        ]
        col_y_range = (cfg.margin, cfg.page_h - cfg.margin)
        grid_positions = None
    else:
        col_x_ranges = None
        col_y_range = None
        grid_positions = None

    grid_idx = 0
    for smi in smiles_pool[: n_structures * 2]:
        size = random.randint(*cfg.struct_size_range)
        struct_img = render_structure(smi, size, cfg)
        if struct_img is None:
            continue

        if use_grid and grid_positions and grid_idx < len(grid_positions):
            gx0, gy0, gx1, gy1 = grid_positions[grid_idx]
            box = place_structure(
                page, struct_img, cfg, existing_boxes,
                x_range=(gx0, gx1), y_range=(gy0, gy1),
            )
            grid_idx += 1
        elif two_column:
            assert col_x_ranges is not None
            col = len(panels) % 2
            box = place_structure(
                page, struct_img, cfg, existing_boxes,
                x_range=col_x_ranges[col], y_range=col_y_range,
            )
        else:
            box = place_structure(page, struct_img, cfg, existing_boxes)
        if box is None:
            continue

        existing_boxes.append(box)

        label_box: Optional[Tuple[int, int, int, int]] = None
        label_text: Optional[str] = None
        # ~10% of structures have no label (e.g. scaffold in a series)
        if random.random() > 0.10:
            label_box, label_text = add_label_near_structure(page, box, cfg, font_paths)
            existing_boxes.append(label_box)

        panels.append({
            "struct_box": box,
            "label_box": label_box,
            "label_text": label_text,
            "smiles": smi,
        })

        if len(panels) >= n_structures:
            break

    # -- Distractors: text elements --
    # Prose blocks: large blocks of body text
    for _ in range(random.randint(3, 7)):
        add_prose_block(page, cfg, font_paths, existing_boxes)
    # Multiline text blocks
    for _ in range(random.randint(5, 10)):
        add_multiline_text_block(page, cfg, font_paths, existing_boxes)
    # Captions
    for _ in range(random.randint(2, 5)):
        add_caption(page, cfg, font_paths, existing_boxes)
    # Stray text snippets
    for _ in range(random.randint(8, 16)):
        add_stray_text(page, cfg, font_paths, existing_boxes)
    # Equation / measurement fragments
    for _ in range(random.randint(4, 8)):
        add_equation_fragment(page, cfg, font_paths, existing_boxes)
    # Section headers
    for _ in range(random.randint(2, 5)):
        add_section_header(page, cfg, font_paths, existing_boxes)
    # Footnotes
    for _ in range(random.randint(1, 4)):
        add_footnote(page, cfg, font_paths, existing_boxes)
    # Journal header
    if random.random() < 0.95:
        add_journal_header(page, cfg, font_paths, existing_boxes)
    # R-group tables
    for _ in range(random.randint(1, 3)):
        add_rgroup_table(page, cfg, font_paths, existing_boxes)
    # Arrows
    for _ in range(random.randint(1, 5)):
        add_arrow(page, cfg, existing_boxes)
    # Panel borders
    for _ in range(random.randint(0, 3)):
        add_panel_border(page, cfg, existing_boxes)
    add_page_number(page, cfg, font_paths, existing_boxes)

    # -- Random images / charts --
    if distractor_pool:
        n_images = random.randint(3, 7)
    else:
        n_images = random.randint(2, 5)
    for _ in range(n_images):
        add_random_image(page, cfg, existing_boxes, distractor_pool)

    return page, panels


def yolo_label(box: Tuple[int, int, int, int], w: int, h: int, class_id: int = 0) -> str:
    """Convert a pixel bounding box to a YOLO-format annotation line.

    YOLO format: <class> <cx> <cy> <bw> <bh>  (all normalised 0-1).
    """
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0 / w   # normalised centre-x
    cy = (y0 + y1) / 2.0 / h   # normalised centre-y
    bw = (x1 - x0) / w          # normalised width
    bh = (y1 - y0) / h          # normalised height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def find_fonts(fonts_dir: Optional[Path]) -> List[Path]:
    if not fonts_dir or not fonts_dir.exists():
        return []
    font_paths = []
    for ext in ("*.ttf", "*.otf"):
        font_paths.extend(fonts_dir.rglob(ext))
    return font_paths


def save_sample(
    page: Image.Image,
    panels: List[dict],
    out_img: Path,
    out_lbl: Path,
    out_gt: Path,
    fmt: str,
    cfg: PageConfig,
    grayscale: bool = False,
) -> None:
    """Post-process and save one generated page.

    1. Apply random noise / JPEG artefacts / blur (data augmentation).
    2. Optionally convert to grayscale (keeps 3-channel for YOLO compat).
    3. Write the image, YOLO label file, and ground-truth JSON.
    """
    # --- Data-augmentation noise pass ---
    page = apply_noise(page, cfg)

    if grayscale:
        # L -> RGB keeps 3 channels so YOLO loaders don't complain.
        page = page.convert("L").convert("RGB")

    # Save the page image in the requested format.
    if fmt.lower() == "jpg":
        page.save(out_img, format="JPEG", quality=random.randint(60, 90))
    else:
        page.save(out_img, format="PNG")

    # --- Build YOLO labels and ground-truth records ---
    yolo_lines = []
    gt_records = []
    for p in panels:
        sb = p["struct_box"]
        lb = p["label_box"]
        # Clamp boxes to page bounds (labels near edges may poke outside).
        struct_box = clamp_box(sb, cfg.page_w, cfg.page_h)
        yolo_lines.append(yolo_label(struct_box, cfg.page_w, cfg.page_h, class_id=0))
        if lb is not None:
            label_box = clamp_box(lb, cfg.page_w, cfg.page_h)
            yolo_lines.append(yolo_label(label_box, cfg.page_w, cfg.page_h, class_id=1))
        else:
            label_box = None
        gt_records.append({
            "struct_bbox": list(struct_box),
            "label_bbox":  list(label_box) if label_box is not None else None,
            "label_text":  p["label_text"],
            "smiles":      p["smiles"],
        })

    # Write YOLO .txt (one line per object) and GT .json.
    out_lbl.write_text("\n".join(yolo_lines))
    out_gt.write_text(json.dumps(gt_records, indent=2))


def generate_dataset(
    smiles_csv: Path,
    out_dir: Path,
    num_train: int,
    num_val: int,
    seed: int,
    fmt: str,
    fonts_dir: Optional[Path],
    distractors_dir: Optional[Path] = None,
    dpi_choices: Optional[List[int]] = None,
    grayscale: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if dpi_choices is None:
        dpi_choices = [300]

    smiles_pool = load_smiles(smiles_csv)
    font_paths = find_fonts(fonts_dir)

    # Load real distractor images if available
    distractor_pool = load_distractor_images(distractors_dir)
    if distractor_pool:
        print(f"Loaded {len(distractor_pool)} distractor images from {distractors_dir}")
    else:
        print("No distractor images found — using only synthetic generators.")

    print(f"DPI choices: {dpi_choices}  |  Grayscale: {grayscale}")

    train_img_dir = out_dir / "train" / "images"
    train_lbl_dir = out_dir / "train" / "labels"
    train_gt_dir  = out_dir / "train" / "ground_truth"
    val_img_dir   = out_dir / "val" / "images"
    val_lbl_dir   = out_dir / "val" / "labels"
    val_gt_dir    = out_dir / "val" / "ground_truth"

    for d in (train_img_dir, train_lbl_dir, train_gt_dir,
              val_img_dir, val_lbl_dir, val_gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    def run_split(count: int, img_dir: Path, lbl_dir: Path, gt_dir: Path, split: str) -> None:
        for i in tqdm(range(count), desc=f"Generating {split}"):
            dpi = random.choice(dpi_choices)
            # 20% of pages are landscape slides (PowerPoint/Keynote PDFs).
            # Cap slide DPI at 200 — slide PDFs are rarely rendered above that.
            if random.random() < 0.20:
                cfg = make_page_config_slide(min(dpi, 200))
            else:
                cfg = make_page_config(dpi)
            # 15% negative pages (no structures) — hard negatives
            if random.random() < 0.15:
                page, panels = make_negative_page(cfg, font_paths, distractor_pool)
            else:
                page, panels = make_page(smiles_pool, cfg, font_paths, distractor_pool)
            img_path = img_dir / f"{split}_{i:06d}.{fmt}"
            lbl_path = lbl_dir / f"{split}_{i:06d}.txt"
            gt_path  = gt_dir  / f"{split}_{i:06d}.json"
            save_sample(page, panels, img_path, lbl_path, gt_path, fmt, cfg, grayscale)

    run_split(num_train, train_img_dir, train_lbl_dir, train_gt_dir, "train")
    run_split(num_val, val_img_dir, val_lbl_dir, val_gt_dir, "val")


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
    parser.add_argument(
        "--distractors-dir",
        type=Path,
        default=None,
        help="Directory of distractor images (downloaded via download_distractor_images.py)",
    )
    parser.add_argument(
        "--dpi",
        default="96,144,200,300",
        help="Comma-separated DPI values to randomly sample per page (default: 96,144,200,300). "
             "Slide pages are capped at 200 DPI internally regardless of this setting.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        default=True,
        help="Convert pages to grayscale before saving (default: True). "
             "Use --no-grayscale to keep colour.",
    )
    parser.add_argument("--no-grayscale", dest="grayscale", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.smiles.exists():
        print(f"SMILES CSV not found: {args.smiles}")
        return 1

    dpi_choices = [int(d.strip()) for d in args.dpi.split(",")]

    generate_dataset(
        smiles_csv=args.smiles,
        out_dir=args.out,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
        fmt=args.format,
        fonts_dir=args.fonts_dir,
        distractors_dir=args.distractors_dir,
        dpi_choices=dpi_choices,
        grayscale=args.grayscale,
    )

    print(f"\nDone. Dataset saved under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
