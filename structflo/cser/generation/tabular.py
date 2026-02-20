"""Tabular page generators: Excel-like tables and compound grid sheets."""

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from structflo.cser.config import PageConfig
from structflo.cser.rendering.chemistry import render_structure
from structflo.cser.rendering.text import draw_rotated_text, load_font, random_label

# ---------------------------------------------------------------------------
# Realistic data-value generators for table cells
# ---------------------------------------------------------------------------


def _rand_bioactivity() -> str:
    style = random.choice(["nM", "uM", "pM"])
    if style == "nM":
        return f"{random.uniform(0.1, 9999):.1f} nM"
    elif style == "uM":
        return f"{random.uniform(0.01, 100):.2f} \u00b5M"
    else:
        return f"{random.uniform(1, 999):.0f} pM"


def _rand_pic50() -> str:
    return f"{random.uniform(4.0, 10.0):.2f}"


def _rand_logp() -> str:
    return f"{random.uniform(-2.0, 7.0):.2f}"


def _rand_mw() -> str:
    return f"{random.uniform(150, 800):.1f}"


def _rand_activity() -> str:
    return random.choice(["Active", "Inactive", "Partial", "++", "+", "-", ">90%", "<10%"])


def _rand_percent() -> str:
    return f"{random.uniform(0, 100):.1f}%"


def _rand_solubility() -> str:
    return f"{random.uniform(0.1, 500):.1f} \u00b5g/mL"


def _rand_project() -> str:
    return random.choice(["PRJ", "PROJ", "SER", "HIT", "LEAD"]) + f"-{random.randint(100, 9999)}"


_COLUMN_GENERATORS: Dict[str, Callable[[], str]] = {
    "IC50 (nM)":    _rand_bioactivity,
    "EC50 (nM)":    _rand_bioactivity,
    "Ki (nM)":      _rand_bioactivity,
    "pIC50":        _rand_pic50,
    "LogP":         _rand_logp,
    "cLogD":        _rand_logp,
    "MW (Da)":      _rand_mw,
    "Activity":     _rand_activity,
    "Inhibition":   _rand_percent,
    "Solubility":   _rand_solubility,
    "Project":      _rand_project,
    "Purity (%)":   _rand_percent,
    "Yield (%)":    _rand_percent,
    "Batch":        lambda: f"BT-{random.randint(1000, 9999)}",
    "Selectivity":  lambda: f"{random.uniform(1, 1000):.0f}x",
}

_EXCEL_TITLES = [
    "Compound Library Results", "SAR Summary Table", "Screening Data",
    "HTS Results", "Compound Activity Table", "Lead Optimisation Data",
    "Assay Results", "Compound Profiling", "Dose-Response Summary",
    "In Vitro ADME Data", "Fragment Screening Hits",
]

_GRID_TITLES = [
    "Compound Library", "SAR Grid", "Lead Series", "HTS Hits",
    "Analogue Set", "Scaffold Exploration", "Fragment Library",
    "Building Blocks", "Active Compounds", "Screening Panel",
    "Diversity Set", "Core Analogues",
]

_HEADER_COLORS = [
    (70, 130, 180), (100, 149, 237), (60, 100, 160),
    (80, 80, 80), (130, 100, 60), (60, 120, 60), (100, 60, 120),
    (40, 100, 120), (120, 40, 60),
]


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


# ---------------------------------------------------------------------------
# Excel-like page
# ---------------------------------------------------------------------------


def make_excel_page(
    smiles_pool: List[str],
    cfg: PageConfig,
    font_paths: List[Path],
    distractor_pool: Optional[List[Image.Image]] = None,
) -> Tuple[Image.Image, List[dict]]:
    """Spreadsheet-style page: structure column + adjacent label column + data columns.

    The label column sits immediately left or right of the structure column.
    Additional columns carry realistic assay/property values (IC50, LogP, etc.).
    """
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    panels: List[dict] = []

    scale = cfg.page_w / 2480.0
    margin = cfg.margin

    # ---- Optional title ----
    title_h = 0
    if random.random() < 0.75:
        title = random.choice(_EXCEL_TITLES)
        t_sz = max(14, int(cfg.page_h * 0.011))
        t_font = load_font(font_paths, t_sz, prefer_bold=True)
        draw.text((margin, margin), title, font=t_font, fill=(20, 20, 20))
        title_h = int(t_sz * 1.8)

    table_x0 = margin
    table_y0 = margin + title_h
    table_x1 = cfg.page_w - margin
    table_y1 = cfg.page_h - margin

    # ---- Structure cell sizing ----
    # Use 65 % of min → 55 % of max of the normal range so structures render
    # crisply. At 300 DPI this gives ≈182–302 px (vs the default 280–550 px),
    # which still leaves room for 8–15 rows on a portrait A4 page.
    s_lo = max(120, int(cfg.struct_size_range[0] * 0.975))
    s_hi = max(s_lo + 45, int(cfg.struct_size_range[1] * 0.825))
    struct_size = random.randint(s_lo, s_hi)

    cell_pad = max(6, int(10 * scale))
    row_h = struct_size + 2 * cell_pad
    header_h = max(22, int(32 * scale))

    # ---- Column layout ----
    struct_col_w = struct_size + 2 * cell_pad
    # Label column must comfortably fit a full compound ID (e.g. "CHEMBL1234567").
    # At 300 DPI ≈ 200 px minimum; scales linearly at lower DPI.
    label_col_w = max(int(struct_col_w * 0.85), int(200 * scale))
    avail_for_data = table_x1 - table_x0 - struct_col_w - label_col_w
    min_data_w = max(1, int(70 * scale))
    n_data_cols = random.randint(2, max(2, min(5, avail_for_data // min_data_w)))
    data_col_w = max(min_data_w, avail_for_data // max(1, n_data_cols))

    col_names = random.sample(list(_COLUMN_GENERATORS.keys()),
                              min(n_data_cols, len(_COLUMN_GENERATORS)))
    col_fns = [_COLUMN_GENERATORS[n] for n in col_names]

    label_on_right = random.random() < 0.5

    if label_on_right:
        col_xs = [
            table_x0,
            table_x0 + struct_col_w,
            *[table_x0 + struct_col_w + label_col_w + i * data_col_w
              for i in range(n_data_cols)],
        ]
        col_ws = [struct_col_w, label_col_w] + [data_col_w] * n_data_cols
        headers = ["Structure", "Compound ID"] + col_names
        struct_ci, label_ci = 0, 1
    else:
        col_xs = [
            table_x0,
            table_x0 + label_col_w,
            *[table_x0 + label_col_w + struct_col_w + i * data_col_w
              for i in range(n_data_cols)],
        ]
        col_ws = [label_col_w, struct_col_w] + [data_col_w] * n_data_cols
        headers = ["Compound ID", "Structure"] + col_names
        struct_ci, label_ci = 1, 0

    # ---- Header row ----
    hdr_bg = random.choice(_HEADER_COLORS)
    draw.rectangle([table_x0, table_y0, table_x1, table_y0 + header_h], fill=hdr_bg)
    hdr_sz = max(8, int(header_h * 0.48))
    hdr_font = load_font(font_paths, hdr_sz, prefer_bold=True)
    for hx, hw, hname in zip(col_xs, col_ws, headers):
        tw, th = _text_size(draw, hname, hdr_font)
        tx = hx + max(cell_pad, (hw - tw) // 2)
        ty = table_y0 + max(2, (header_h - th) // 2)
        draw.text((tx, ty), hname, font=hdr_font, fill=(255, 255, 255))

    # ---- Per-row fonts ----
    cell_sz = max(8, int(row_h * 0.12))
    cell_font = load_font(font_paths, cell_sz)
    lbl_sz = max(8, min(cell_sz + 2, int(label_col_w * 0.11)))
    lbl_font = load_font(font_paths, lbl_sz)

    alt_bg = (245, 245, 252)  # very-light alternating tint

    y = table_y0 + header_h
    row_idx = 0

    for smi in smiles_pool:
        if y + row_h > table_y1:
            break

        struct_img = render_structure(smi, struct_size, cfg)
        if struct_img is None:
            continue

        # Alternating row background
        if row_idx % 2 == 1:
            draw.rectangle([table_x0, y, table_x1, y + row_h], fill=alt_bg)

        # Structure
        sw, sh = struct_img.size
        sx = col_xs[struct_ci] + max(cell_pad, (col_ws[struct_ci] - sw) // 2)
        sy = y + max(cell_pad, (row_h - sh) // 2)
        page.paste(struct_img, (sx, sy), struct_img)
        struct_box = (sx, sy, sx + sw, sy + sh)

        # Label (centred in its cell, no rotation in spreadsheet context)
        label = random_label()
        lcx = col_xs[label_ci] + col_ws[label_ci] // 2
        lcy = y + row_h // 2
        label_box = draw_rotated_text(page, label, (lcx, lcy), lbl_font, 0.0)

        # Data cells
        for di, (dx, fn) in enumerate(zip(col_xs[2:], col_fns)):
            val = fn()
            dw = col_ws[di + 2]
            vw, vh = _text_size(draw, val, cell_font)
            draw.text(
                (dx + max(cell_pad, (dw - vw) // 2), y + max(2, (row_h - vh) // 2)),
                val, font=cell_font, fill=(30, 30, 30),
            )

        # Row separator
        draw.line([(table_x0, y + row_h), (table_x1, y + row_h)],
                  fill=(200, 200, 200), width=1)

        panels.append({
            "struct_box": struct_box,
            "label_box":  label_box,
            "label_text": label,
            "smiles":     smi,
        })
        y += row_h
        row_idx += 1

    # ---- Column separators ----
    table_bottom = y
    for cx in col_xs:
        draw.line([(cx, table_y0), (cx, table_bottom)], fill=(180, 180, 180), width=1)
    draw.line(
        [(col_xs[-1] + col_ws[-1], table_y0),
         (col_xs[-1] + col_ws[-1], table_bottom)],
        fill=(180, 180, 180), width=1,
    )
    # Outer border
    draw.rectangle([table_x0, table_y0, table_x1, table_bottom],
                   outline=(120, 120, 120), width=2)

    return page, panels


# ---------------------------------------------------------------------------
# Compound grid page
# ---------------------------------------------------------------------------


def make_grid_page(
    smiles_pool: List[str],
    cfg: PageConfig,
    font_paths: List[Path],
    distractor_pool: Optional[List[Image.Image]] = None,
) -> Tuple[Image.Image, List[dict]]:
    """Clean compound-grid page: structures in a regular grid with labels above/below.

    Labels sit uniformly above or below every structure.  Optionally a few
    property values (MW, IC50, …) are printed alongside the label.  Thin grid
    lines may or may not be drawn.
    """
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    panels: List[dict] = []

    scale = cfg.page_w / 2480.0
    margin = cfg.margin

    # ---- Optional title ----
    start_y = margin
    if random.random() < 0.8:
        title = random.choice(_GRID_TITLES)
        t_sz = max(14, int(cfg.page_h * 0.013))
        t_font = load_font(font_paths, t_sz, prefer_bold=True)
        draw.text((margin, start_y), title, font=t_font, fill=(20, 20, 20))
        start_y += int(t_sz * 2.0)

    # ---- Grid layout ----
    cols = random.choice([2, 3, 4, 5])
    usable_w = cfg.page_w - 2 * margin
    cell_w = usable_w // cols

    struct_size = int(cell_w * random.uniform(0.55, 0.72))
    struct_size = max(int(55 * scale), struct_size)

    label_above = random.random() < 0.5

    # Optional extra data rows beneath the label
    n_extra = random.randint(0, 3)
    extra_names = random.sample(list(_COLUMN_GENERATORS.keys()),
                                min(n_extra, len(_COLUMN_GENERATORS)))
    extra_fns = [_COLUMN_GENERATORS[n] for n in extra_names]

    lbl_sz = max(8, int(struct_size * 0.09))
    lbl_font = load_font(font_paths, lbl_sz)
    data_sz = max(7, int(struct_size * 0.075))
    data_font = load_font(font_paths, data_sz)

    label_h = int(lbl_sz * 1.8)
    data_line_h = int(data_sz * 1.5)
    data_h = n_extra * data_line_h
    struct_pad = max(int(8 * scale), 6)

    cell_h = struct_size + label_h + data_h + 3 * struct_pad
    usable_h = cfg.page_h - start_y - margin
    rows = max(1, usable_h // cell_h)

    # ---- Optional thin grid lines ----
    if random.random() < 0.45:
        gc = (210, 210, 210)
        for r in range(rows + 1):
            gy = start_y + r * cell_h
            draw.line([(margin, gy), (margin + cols * cell_w, gy)], fill=gc, width=1)
        for c in range(cols + 1):
            gx = margin + c * cell_w
            draw.line([(gx, start_y), (gx, start_y + rows * cell_h)], fill=gc, width=1)

    smi_list = smiles_pool[:]
    random.shuffle(smi_list)
    smi_iter = iter(smi_list)

    for row in range(rows):
        for col in range(cols):
            smi = next(smi_iter, None)
            if smi is None:
                break

            struct_img = render_structure(smi, struct_size, cfg)
            if struct_img is None:
                continue

            sw, sh = struct_img.size
            cell_x = margin + col * cell_w
            cell_y = start_y + row * cell_h

            # Structure Y: shifted down if label + data go above
            if label_above:
                sy = cell_y + struct_pad + label_h + data_h
            else:
                sy = cell_y + struct_pad

            sx = cell_x + (cell_w - sw) // 2
            page.paste(struct_img, (sx, sy), struct_img)
            struct_box = (sx, sy, sx + sw, sy + sh)

            # Label
            label = random_label()
            label_cx = cell_x + cell_w // 2
            if label_above:
                label_cy = cell_y + struct_pad + label_h // 2
            else:
                label_cy = sy + sh + struct_pad + label_h // 2

            label_box = draw_rotated_text(page, label, (label_cx, label_cy), lbl_font, 0.0)

            # Extra property rows
            for di, (dname, dfn) in enumerate(zip(extra_names, extra_fns)):
                val_str = f"{dname}: {dfn()}"
                if label_above:
                    dy = cell_y + struct_pad + label_h + di * data_line_h
                else:
                    dy = label_cy + label_h // 2 + struct_pad // 2 + di * data_line_h
                dw, _ = _text_size(draw, val_str, data_font)
                dx = cell_x + max(4, (cell_w - dw) // 2)
                draw.text((dx, dy), val_str, font=data_font, fill=(80, 80, 80))

            panels.append({
                "struct_box": struct_box,
                "label_box":  label_box,
                "label_text": label,
                "smiles":     smi,
            })

    return page, panels
