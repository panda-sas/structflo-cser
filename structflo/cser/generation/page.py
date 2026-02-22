"""Page assembly: make_page(), make_negative_page(), apply_noise()."""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from structflo.cser.config import PageConfig
from structflo.cser.distractors import (
    add_arrow,
    add_caption,
    add_equation_fragment,
    add_footnote,
    add_journal_header,
    add_multiline_text_block,
    add_page_number,
    add_panel_border,
    add_prose_block,
    add_random_image,
    add_rgroup_table,
    add_section_header,
    add_stray_text,
)
from structflo.cser.rendering.chemistry import _to_dark_mode, place_structure, render_structure
from structflo.cser.rendering.text import add_label_near_structure


def _page_bg_color(cfg: PageConfig) -> Tuple[int, int, int]:
    """Return a background colour for the page canvas.

    Pure white (72%) / cream aged-paper (8%) / light-grey (4%) — remaining
    probability is pure white again so it stays the dominant case.
    """
    roll = random.random()
    if roll < cfg.page_bg_tint_prob * 0.67:
        # Cream / aged paper (common in scanned journal pages and patents)
        return (
            random.randint(248, 254),
            random.randint(240, 250),
            random.randint(218, 238),
        )
    if roll < cfg.page_bg_tint_prob:
        # Very light grey (common in LaTeX-generated PDFs viewed on screen)
        v = random.randint(238, 250)
        return (v, v, v)
    return (255, 255, 255)


def apply_noise(img: Image.Image, cfg: PageConfig) -> Image.Image:
    """Apply random photometric degradations to a page image.

    Models real-world document imperfections:
    - Brightness jitter      — scanner exposure / print darkness variation
    - Contrast jitter        — gamma / tone-curve differences across devices
    - Scanner lamp gradient  — non-uniform illumination across the scan bed
    - Gaussian blur          — flatbed scanner softness / slight defocus
    - Gaussian noise         — scanner CCD noise
    """
    out = img

    # Brightness: multiplicative exposure shift
    if random.random() < cfg.brightness_prob:
        factor = random.uniform(0.80, 1.20)
        arr = np.array(out).astype(np.float32)
        arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

    # Contrast: stretch / compress around mid-grey (gamma-like)
    if random.random() < 0.30:
        arr = np.array(out).astype(np.float32)
        c = random.uniform(0.80, 1.25)
        arr = np.clip((arr - 128.0) * c + 128.0, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

    # Scanner lamp falloff: a smooth brightness gradient across the page,
    # brighter on one side — common in flatbed scans of bound documents.
    if random.random() < 0.10:
        arr = np.array(out).astype(np.float32)
        h, w = arr.shape[:2]
        strength = random.uniform(0.04, 0.10)
        side = random.choice(["left", "right", "top", "bottom"])
        if side == "left":
            grad = np.linspace(1 - strength, 1 + strength, w)[np.newaxis, :, np.newaxis]
        elif side == "right":
            grad = np.linspace(1 + strength, 1 - strength, w)[np.newaxis, :, np.newaxis]
        elif side == "top":
            grad = np.linspace(1 - strength, 1 + strength, h)[:, np.newaxis, np.newaxis]
        else:
            grad = np.linspace(1 + strength, 1 - strength, h)[:, np.newaxis, np.newaxis]
        arr = np.clip(arr * grad, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

    if random.random() < cfg.blur_prob:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))

    if random.random() < cfg.noise_prob:
        arr = np.array(out).astype(np.int16)
        noise = np.random.normal(0, 6, size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)

    return out


def make_negative_page(
    cfg: PageConfig,
    font_paths: List[Path],
    distractor_pool: Optional[List[Image.Image]] = None,
) -> Tuple[Image.Image, List[dict]]:
    """Generate a page with no chemical structures — pure text/charts.

    Used as hard negatives so the model learns to output nothing when
    there are no structures present.
    """
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), _page_bg_color(cfg))
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

    n_images = random.randint(2, 5) if distractor_pool else random.randint(1, 4)
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
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), _page_bg_color(cfg))

    existing_boxes: List[Tuple[int, int, int, int]] = []
    panels: List[dict] = []

    n_structures = random.randint(cfg.min_structures, cfg.max_structures)
    random.shuffle(smiles_pool)

    # Choose layout mode
    layout_roll = random.random()
    use_grid = layout_roll < cfg.grid_layout_prob and n_structures >= 4
    two_column = (
        (not use_grid)
        and layout_roll < cfg.grid_layout_prob + cfg.two_column_prob
        and n_structures >= 2
    )

    if use_grid:
        cols = random.choice([2, 3, 4])
        rows = (n_structures + cols - 1) // cols
        usable_w = cfg.page_w - 2 * cfg.margin
        usable_h = cfg.page_h - 2 * cfg.margin
        cell_w = usable_w // cols
        cell_h = usable_h // rows
        grid_positions = [
            (
                cfg.margin + c * cell_w,
                cfg.margin + r * cell_h,
                cfg.margin + (c + 1) * cell_w,
                cfg.margin + (r + 1) * cell_h,
            )
            for r in range(rows)
            for c in range(cols)
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

    _DARK_BG_COLORS = [
        (15, 15, 15), (30, 30, 30), (20, 20, 55),
        (35, 10, 10), (10, 35, 10), (40, 30, 0),
    ]

    grid_idx = 0
    for smi in smiles_pool[: n_structures * 2]:
        size = random.randint(*cfg.struct_size_range)
        struct_img = render_structure(smi, size, cfg)
        if struct_img is None:
            continue

        # Dark-background variant: ~15% of structures rendered on a dark patch.
        # Models highlighted compound boxes in slides and coloured-background PDFs.
        # Bonds are inverted to remain visible; the page gets a filled dark rect.
        if random.random() < cfg.dark_bg_prob:
            struct_img = _to_dark_mode(struct_img)
            dark_color: Optional[Tuple[int, int, int]] = random.choice(_DARK_BG_COLORS)
        else:
            dark_color = None

        if use_grid and grid_positions and grid_idx < len(grid_positions):
            gx0, gy0, gx1, gy1 = grid_positions[grid_idx]
            box = place_structure(
                page,
                struct_img,
                cfg,
                existing_boxes,
                x_range=(gx0, gx1),
                y_range=(gy0, gy1),
                dark_bg=dark_color,
            )
            grid_idx += 1
        elif two_column:
            assert col_x_ranges is not None
            col = len(panels) % 2
            box = place_structure(
                page,
                struct_img,
                cfg,
                existing_boxes,
                x_range=col_x_ranges[col],
                y_range=col_y_range,
                dark_bg=dark_color,
            )
        else:
            box = place_structure(page, struct_img, cfg, existing_boxes, dark_bg=dark_color)

        if box is None:
            continue

        existing_boxes.append(box)

        label_box: Optional[Tuple[int, int, int, int]] = None
        label_text: Optional[str] = None
        if random.random() > 0.10:  # ~10% of structures have no label
            label_box, label_text = add_label_near_structure(page, box, cfg, font_paths)
            existing_boxes.append(label_box)

        panels.append(
            {
                "struct_box": box,
                "label_box": label_box,
                "label_text": label_text,
                "smiles": smi,
            }
        )

        if len(panels) >= n_structures:
            break

    # Distractors: text elements
    for _ in range(random.randint(3, 7)):
        add_prose_block(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(5, 10)):
        add_multiline_text_block(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(2, 5)):
        add_caption(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(8, 16)):
        add_stray_text(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(4, 8)):
        add_equation_fragment(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(2, 5)):
        add_section_header(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(1, 4)):
        add_footnote(page, cfg, font_paths, existing_boxes)
    if random.random() < 0.95:
        add_journal_header(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(1, 3)):
        add_rgroup_table(page, cfg, font_paths, existing_boxes)
    for _ in range(random.randint(1, 5)):
        add_arrow(page, cfg, existing_boxes)
    for _ in range(random.randint(0, 3)):
        add_panel_border(page, cfg, existing_boxes)
    add_page_number(page, cfg, font_paths, existing_boxes)

    # Random images / charts
    n_images = random.randint(3, 7) if distractor_pool else random.randint(2, 5)
    for _ in range(n_images):
        add_random_image(page, cfg, existing_boxes, distractor_pool)

    return page, panels
