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
from structflo.cser.rendering.chemistry import place_structure, render_structure
from structflo.cser.rendering.text import add_label_near_structure


def apply_noise(img: Image.Image, cfg: PageConfig) -> Image.Image:
    """Apply random brightness jitter, blur, and Gaussian noise to a page."""
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
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), (255, 255, 255))

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

    grid_idx = 0
    for smi in smiles_pool[: n_structures * 2]:
        size = random.randint(*cfg.struct_size_range)
        struct_img = render_structure(smi, size, cfg)
        if struct_img is None:
            continue

        if use_grid and grid_positions and grid_idx < len(grid_positions):
            gx0, gy0, gx1, gy1 = grid_positions[grid_idx]
            box = place_structure(
                page,
                struct_img,
                cfg,
                existing_boxes,
                x_range=(gx0, gx1),
                y_range=(gy0, gy1),
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
            )
        else:
            box = place_structure(page, struct_img, cfg, existing_boxes)

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
