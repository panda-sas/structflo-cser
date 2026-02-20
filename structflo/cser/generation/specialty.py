"""Specialty page generators: data cards, SAR R-group tables, MMP sheets."""

import random
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from structflo.cser.config import PageConfig
from structflo.cser.generation.tabular import _COLUMN_GENERATORS, _text_size
from structflo.cser.rendering.chemistry import render_structure
from structflo.cser.rendering.text import draw_rotated_text, load_font, random_label

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_R_SUBSTITUENTS = [
    "-H", "-F", "-Cl", "-Br", "-I",
    "-CH\u2083", "-OMe", "-OEt", "-CF\u2083", "-CHF\u2082",
    "-CN", "-NH\u2082", "-OH", "-SMe", "-SO\u2082Me",
    "-iPr", "-tBu", "-Ph", "-Bn",
    "4-F-Ph", "3-Cl-Ph", "2-Me-Ph", "-cyclopropyl",
]

_TRANSFORMATIONS = [
    "-H \u2192 -F", "-H \u2192 -CH\u2083", "-H \u2192 -OMe",
    "-CH\u2083 \u2192 -CF\u2083", "-H \u2192 -Cl", "-OMe \u2192 -OEt",
    "-Me \u2192 -Et", "-H \u2192 -CN", "-Cl \u2192 -F",
    "-NH\u2082 \u2192 -NHMe", "-H \u2192 -Br", "-OH \u2192 -OMe",
    "-H \u2192 -SO\u2082Me", "-CH\u2083 \u2192 -CF\u2082H",
]

_CARD_TITLES = [
    "COMPOUND REGISTRATION", "COMPOUND DATA SHEET",
    "COMPOUND RECORD", "STRUCTURE REPORT",
    "COMPOUND PROFILE", "COMPOUND SUMMARY",
]

_CARD_PROPS = {
    "MW":              lambda: f"{random.uniform(150, 800):.1f}",
    "Formula":         lambda: (
        f"C{random.randint(8,28)}H{random.randint(8,38)}"
        f"N{random.randint(0,4)}O{random.randint(0,5)}"
    ),
    "cLogP":           lambda: f"{random.uniform(-2, 7):.2f}",
    "TPSA (\u00c5\u00b2)": lambda: f"{random.uniform(20, 160):.1f}",
    "HBD":             lambda: str(random.randint(0, 5)),
    "HBA":             lambda: str(random.randint(0, 10)),
    "Purity (%)":      lambda: f"{random.uniform(85, 99.9):.1f}",
    "Batch":           lambda: f"BT-{random.randint(1000, 9999)}",
    "Project":         lambda: (
        random.choice(["PRJ", "PROJ", "SER", "HIT"]) + f"-{random.randint(100, 9999)}"
    ),
    "Source":          lambda: random.choice(
        ["Internal", "Commercial", "Outsourced", "Synthesised"]
    ),
    "Storage":         lambda: random.choice(["-20\u00b0C", "-80\u00b0C", "RT", "4\u00b0C"]),
    "Solvent":         lambda: random.choice(["DMSO", "MeOH", "Water", "ACN"]),
    "Conc. (mM)":      lambda: f"{random.uniform(1, 100):.1f}",
    "Vol. (\u00b5L)":  lambda: f"{random.uniform(50, 500):.0f}",
}

_SAR_TITLES = [
    "Structure-Activity Relationship Summary",
    "SAR Table", "Compound Series Overview",
    "Lead Optimisation: Analogue Series",
    "In Vitro SAR Data", "Medicinal Chemistry Summary",
]

_MMP_TITLES = [
    "Matched Molecular Pair Analysis",
    "Bioisostere Comparison", "SAR by MMP",
    "Structural Transformations", "Activity Cliff Analysis",
]

_HDR_COLORS = [
    (40, 60, 100), (60, 40, 100), (40, 90, 60),
    (100, 50, 30), (50, 80, 100), (80, 60, 40),
    (30, 80, 90),
]


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    x0: int, y: int, x1: int,
    color=(80, 80, 80),
    width: int = 3,
    head: int = 12,
) -> None:
    """Draw a horizontal right-pointing arrow."""
    draw.line([(x0, y), (x1 - head, y)], fill=color, width=width)
    draw.polygon(
        [(x1, y), (x1 - head, y - head // 2), (x1 - head, y + head // 2)],
        fill=color,
    )


# ---------------------------------------------------------------------------
# Single-compound data card
# ---------------------------------------------------------------------------


def make_data_card_page(
    smiles_pool: List[str],
    cfg: PageConfig,
    font_paths: List[Path],
    distractor_pool: Optional[List[Image.Image]] = None,
) -> Tuple[Image.Image, List[dict]]:
    """Compound registration card: one large structure, prominent ID, property grid."""
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), (250, 250, 252))
    draw = ImageDraw.Draw(page)
    scale = cfg.page_w / 2480.0
    margin = cfg.margin

    # ---- Coloured header bar ----
    hdr_h = max(int(58 * scale), 30)
    hdr_color = random.choice(_HDR_COLORS)
    draw.rectangle([0, 0, cfg.page_w, hdr_h], fill=hdr_color)
    t_sz = max(14, int(22 * scale))
    t_font = load_font(font_paths, t_sz, prefer_bold=True)
    title_str = random.choice(_CARD_TITLES)
    _, th = _text_size(draw, title_str, t_font)
    draw.text((margin, (hdr_h - th) // 2), title_str, font=t_font, fill=(255, 255, 255))

    # ---- Large structure (upper 56 % of remaining space) ----
    avail_h = cfg.page_h - hdr_h - 2 * margin
    struct_area_h = int(avail_h * 0.56)
    struct_size = min(
        cfg.struct_size_range[1],
        min(cfg.page_w - 2 * margin, struct_area_h),
    )

    struct_img, smi = None, None
    for s in smiles_pool:
        img = render_structure(s, struct_size, cfg)
        if img is not None:
            struct_img, smi = img, s
            break

    if struct_img is None:
        return page, []

    sw, sh = struct_img.size
    sx = (cfg.page_w - sw) // 2
    sy = hdr_h + margin + (struct_area_h - sh) // 2
    page.paste(struct_img, (sx, sy), struct_img)
    struct_box = (sx, sy, sx + sw, sy + sh)

    # ---- Compound ID — large, centred below structure ----
    label = random_label()
    id_sz = max(int(28 * scale), int(sw * 0.09))
    id_font = load_font(font_paths, id_sz, prefer_bold=True)

    # Approximate vertical centre for draw_rotated_text (it centres at cy).
    # Use id_sz as a rough estimate of half the rendered height.
    label_cx = cfg.page_w // 2
    label_cy = sy + sh + max(int(20 * scale), 12) + id_sz

    style = random.choice(["box", "underline", "plain", "colored_text"])
    fill = hdr_color if style == "colored_text" else (20, 20, 20)

    # draw_rotated_text pixel-scans the glyphs → correct tight bbox regardless
    # of font bearing offsets.
    label_box = draw_rotated_text(page, label, (label_cx, label_cy), id_font, 0.0,
                                  fill=fill)

    # Post-render decorations (drawn over text edges, which is fine for outlines)
    bp = max(int(8 * scale), 4)
    if style == "box":
        draw.rectangle(
            [label_box[0] - bp, label_box[1] - bp,
             label_box[2] + bp, label_box[3] + bp],
            outline=hdr_color, width=max(1, int(2 * scale)),
        )
    elif style == "underline":
        uy = label_box[3] + max(int(4 * scale), 2)
        draw.line([(label_box[0], uy), (label_box[2], uy)],
                  fill=hdr_color, width=max(1, int(2 * scale)))

    # ---- Horizontal separator ----
    sep_y = label_box[3] + int(22 * scale)
    draw.line(
        [(margin, sep_y), (cfg.page_w - margin, sep_y)],
        fill=(190, 190, 190), width=max(1, int(1.5 * scale)),
    )

    # ---- Two-column property grid ----
    py = sep_y + int(18 * scale)
    prop_keys = random.sample(list(_CARD_PROPS.keys()), min(12, len(_CARD_PROPS)))
    prop_sz = max(10, int(18 * scale))
    prop_font = load_font(font_paths, prop_sz)
    bold_font = load_font(font_paths, prop_sz, prefer_bold=True)
    col_w = (cfg.page_w - 2 * margin) // 2
    line_h = max(int(prop_sz * 1.8), int(30 * scale))

    for i, key in enumerate(prop_keys):
        val = _CARD_PROPS[key]()
        kx = margin + (i % 2) * col_w
        ky = py + (i // 2) * line_h
        key_str = f"{key}:"
        draw.text((kx, ky), key_str, font=bold_font, fill=(80, 80, 80))
        kw, _ = _text_size(draw, key_str, bold_font)
        draw.text((kx + kw + max(int(8 * scale), 4), ky), val,
                  font=prop_font, fill=(20, 20, 20))

    return page, [{
        "struct_box": struct_box,
        "label_box":  label_box,
        "label_text": label,
        "smiles":     smi,
    }]


# ---------------------------------------------------------------------------
# SAR R-group table
# ---------------------------------------------------------------------------


def make_sar_page(
    smiles_pool: List[str],
    cfg: PageConfig,
    font_paths: List[Path],
    distractor_pool: Optional[List[Image.Image]] = None,
) -> Tuple[Image.Image, List[dict]]:
    """SAR page: scaffold at top + analogue table with compound IDs and R-group annotations.

    Each row is annotated as a (structure, compound_ID) pair.  The scaffold
    itself is paired with a series-level compound ID.  R-group columns contain
    plain substituent text (-CH₃, -OMe, …) — NOT annotated as labels.
    """
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    scale = cfg.page_w / 2480.0
    margin = cfg.margin
    panels: List[dict] = []

    # ---- Title ----
    t_sz = max(14, int(cfg.page_h * 0.010))
    t_font = load_font(font_paths, t_sz, prefer_bold=True)
    draw.text((margin, margin), random.choice(_SAR_TITLES), font=t_font, fill=(20, 20, 20))
    y_cursor = margin + int(t_sz * 2.2)

    # ---- Scaffold section (top ~25 % of remaining vertical space) ----
    scaffold_area_h = int((cfg.page_h - y_cursor - margin) * 0.25)
    scaffold_size = min(
        min(cfg.page_w // 2 - 2 * margin, scaffold_area_h),
        cfg.struct_size_range[1],
    )
    scaffold_size = max(max(80, int(80 * scale)), scaffold_size)

    scaffold_img, scaffold_smi = None, None
    for s in smiles_pool:
        img = render_structure(s, scaffold_size, cfg)
        if img is not None:
            scaffold_img, scaffold_smi = img, s
            break

    if scaffold_img is not None:
        sw, sh = scaffold_img.size
        sx = margin + int(20 * scale)
        sy = y_cursor + max(0, (scaffold_area_h - sh) // 2)
        page.paste(scaffold_img, (sx, sy), scaffold_img)

        core_sz = max(9, int(t_sz * 0.88))
        core_font_b = load_font(font_paths, core_sz, prefer_bold=True)
        core_font = load_font(font_paths, core_sz)
        rx = sx + sw + int(40 * scale)

        draw.text((rx, sy), "Core scaffold:", font=core_font_b, fill=(60, 60, 60))
        sc_label = random_label()
        sly = sy + int(core_sz * 1.9)
        # Centre draw_rotated_text at the left-align equivalent start + half-width
        sc_meta = draw.textbbox((0, 0), sc_label, font=core_font_b)
        sc_cx = rx + (sc_meta[2] - sc_meta[0]) // 2
        sc_cy = sly + (sc_meta[3] - sc_meta[1]) // 2
        sc_label_box = draw_rotated_text(page, sc_label, (sc_cx, sc_cy),
                                         core_font_b, 0.0, fill=(30, 30, 120))

        panels.append({
            "struct_box": (sx, sy, sx + sw, sy + sh),
            "label_box":  sc_label_box,
            "label_text": sc_label,
            "smiles":     scaffold_smi,
        })

    y_cursor += scaffold_area_h + int(12 * scale)
    draw.line(
        [(margin, y_cursor), (cfg.page_w - margin, y_cursor)],
        fill=(140, 140, 140), width=max(1, int(1.5 * scale)),
    )
    y_cursor += int(15 * scale)

    # ---- Table layout ----
    n_r_groups = random.randint(1, 3)
    n_act_cols = random.randint(1, 3)
    act_names = random.sample(list(_COLUMN_GENERATORS.keys()),
                              min(n_act_cols, len(_COLUMN_GENERATORS)))
    act_fns = [_COLUMN_GENERATORS[n] for n in act_names]

    # Struct size for table rows
    s_lo = max(80, int(cfg.struct_size_range[0] * 0.65))
    s_hi = max(s_lo + 30, int(cfg.struct_size_range[1] * 0.55))
    struct_size = random.randint(s_lo, s_hi)
    cell_pad = max(5, int(8 * scale))
    row_h = struct_size + 2 * cell_pad
    hdr_h = max(20, int(28 * scale))

    tbl_w = cfg.page_w - 2 * margin
    id_col_w = max(int(140 * scale), int(tbl_w * 0.14))
    struct_col_w = struct_size + 2 * cell_pad
    rg_col_w = max(int(90 * scale), int(tbl_w * 0.10))
    remaining = tbl_w - id_col_w - struct_col_w - n_r_groups * rg_col_w
    act_col_w = max(int(80 * scale), remaining // max(1, n_act_cols))

    # Column x-positions: ID | Structure | R₁ … Rₙ | Act₁ … Actₙ
    col_xs: List[int] = []
    cx = margin
    col_xs.append(cx); cx += id_col_w
    col_xs.append(cx); cx += struct_col_w
    for _ in range(n_r_groups):
        col_xs.append(cx); cx += rg_col_w
    for _ in range(n_act_cols):
        col_xs.append(cx); cx += act_col_w

    r_labels = [f"R\u2081" if i == 0 else f"R\u2082" if i == 1 else f"R{i+1}"
                for i in range(n_r_groups)]
    col_hdrs = ["Compound", "Structure"] + r_labels + act_names

    # Academic-style header: bold text + bottom border (no colour fill)
    hdr_sz = max(8, int(hdr_h * 0.50))
    hdr_font = load_font(font_paths, hdr_sz, prefer_bold=True)
    for hx, hname in zip(col_xs, col_hdrs):
        draw.text((hx + cell_pad, y_cursor + (hdr_h - hdr_sz) // 2),
                  hname, font=hdr_font, fill=(30, 30, 30))
    draw.line(
        [(margin, y_cursor + hdr_h), (cfg.page_w - margin, y_cursor + hdr_h)],
        fill=(100, 100, 100), width=max(1, int(1.5 * scale)),
    )
    y_cursor += hdr_h

    # ---- Data rows ----
    cell_sz = max(8, int(row_h * 0.11))
    cell_font = load_font(font_paths, cell_sz)
    id_font_sz = max(8, int(id_col_w * 0.10))
    id_font = load_font(font_paths, id_font_sz)

    used_smis = {scaffold_smi} if scaffold_smi else set()
    smi_list = [s for s in smiles_pool if s not in used_smis]
    random.shuffle(smi_list)

    for smi in smi_list:
        if y_cursor + row_h > cfg.page_h - margin:
            break

        struct_img = render_structure(smi, struct_size, cfg)
        if struct_img is None:
            continue

        # Compound ID — centred vertically in the ID column
        label = random_label()
        lcx = col_xs[0] + id_col_w // 2
        lcy = y_cursor + row_h // 2
        label_box = draw_rotated_text(page, label, (lcx, lcy), id_font, 0.0)

        # Structure
        sw, sh = struct_img.size
        sx = col_xs[1] + max(cell_pad, (struct_col_w - sw) // 2)
        sy = y_cursor + max(cell_pad, (row_h - sh) // 2)
        page.paste(struct_img, (sx, sy), struct_img)

        # R-group text
        for ri in range(n_r_groups):
            rval = random.choice(_R_SUBSTITUENTS)
            rx = col_xs[2 + ri]
            rw, _ = _text_size(draw, rval, cell_font)
            draw.text(
                (rx + max(cell_pad, (rg_col_w - rw) // 2),
                 y_cursor + (row_h - cell_sz) // 2),
                rval, font=cell_font, fill=(40, 40, 40),
            )

        # Activity values
        for ai, (ax, fn) in enumerate(zip(col_xs[2 + n_r_groups:], act_fns)):
            val = fn()
            vw, _ = _text_size(draw, val, cell_font)
            draw.text(
                (ax + max(cell_pad, (act_col_w - vw) // 2),
                 y_cursor + (row_h - cell_sz) // 2),
                val, font=cell_font, fill=(30, 30, 30),
            )

        # Thin row separator
        draw.line(
            [(margin, y_cursor + row_h), (cfg.page_w - margin, y_cursor + row_h)],
            fill=(210, 210, 210), width=1,
        )

        panels.append({
            "struct_box": (sx, sy, sx + sw, sy + sh),
            "label_box":  label_box,
            "label_text": label,
            "smiles":     smi,
        })
        y_cursor += row_h

    draw.line(
        [(margin, y_cursor), (cfg.page_w - margin, y_cursor)],
        fill=(100, 100, 100), width=max(1, int(1.5 * scale)),
    )
    return page, panels


# ---------------------------------------------------------------------------
# Matched molecular pair (MMP) sheet
# ---------------------------------------------------------------------------


def make_mmp_page(
    smiles_pool: List[str],
    cfg: PageConfig,
    font_paths: List[Path],
    distractor_pool: Optional[List[Image.Image]] = None,
) -> Tuple[Image.Image, List[dict]]:
    """MMP sheet: stacked pairs of structures connected by a transformation arrow.

    Each structure is annotated with a compound ID (catalog-style, never a bare
    number like "1" or "2a").  Transformation text and Δ-activity values between
    the two structures are drawn but NOT annotated.
    """
    page = Image.new("RGB", (cfg.page_w, cfg.page_h), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    scale = cfg.page_w / 2480.0
    margin = cfg.margin
    panels: List[dict] = []

    # ---- Title ----
    t_sz = max(14, int(cfg.page_h * 0.010))
    t_font = load_font(font_paths, t_sz, prefer_bold=True)
    draw.text((margin, margin), random.choice(_MMP_TITLES), font=t_font, fill=(20, 20, 20))
    y_cursor = margin + int(t_sz * 2.2)

    # ---- Column headers ----
    usable_w = cfg.page_w - 2 * margin
    struct_panel_w = int(usable_w * 0.38)
    arrow_panel_w = usable_w - 2 * struct_panel_w

    hdr_sz = max(9, int(t_sz * 0.75))
    hdr_font = load_font(font_paths, hdr_sz, prefer_bold=True)
    col_labels = [
        ("Compound A",     margin + struct_panel_w // 2),
        ("Transformation", margin + struct_panel_w + arrow_panel_w // 2),
        ("Compound B",     margin + struct_panel_w + arrow_panel_w + struct_panel_w // 2),
    ]
    for txt, cx in col_labels:
        tw, _ = _text_size(draw, txt, hdr_font)
        draw.text((cx - tw // 2, y_cursor), txt, font=hdr_font, fill=(80, 80, 80))

    y_cursor += int(hdr_sz * 1.8)
    draw.line(
        [(margin, y_cursor), (cfg.page_w - margin, y_cursor)],
        fill=(160, 160, 160), width=max(1, int(1 * scale)),
    )
    y_cursor += int(12 * scale)

    # ---- Sizing ----
    struct_size = min(
        int(struct_panel_w * 0.72),
        int(cfg.struct_size_range[1] * 0.65),
    )
    struct_size = max(int(80 * scale), struct_size)

    id_sz = max(8, int(struct_size * 0.085))
    id_font = load_font(font_paths, id_sz)
    ann_sz = max(8, int(arrow_panel_w * 0.065))
    ann_font = load_font(font_paths, ann_sz)
    delta_sz = max(7, int(struct_size * 0.075))
    delta_font = load_font(font_paths, delta_sz)

    id_h = int(id_sz * 1.8)
    delta_sz_h = int(delta_sz * 1.5)
    n_delta = random.randint(0, 2)
    delta_names = random.sample(list(_COLUMN_GENERATORS.keys()),
                                min(n_delta, len(_COLUMN_GENERATORS)))
    delta_fns = [_COLUMN_GENERATORS[n] for n in delta_names]

    label_above = random.random() < 0.35  # labels sit below structures most of the time
    struct_pad = int(12 * scale)
    pair_h = struct_size + id_h + n_delta * delta_sz_h + 2 * struct_pad

    # ---- Pairs ----
    smi_list = smiles_pool[:]
    random.shuffle(smi_list)
    smi_iter = iter(smi_list)

    n_pairs = random.randint(2, 5)

    for _ in range(n_pairs):
        if y_cursor + pair_h > cfg.page_h - margin:
            break

        smi_a = next(smi_iter, None)
        smi_b = next(smi_iter, None)
        if smi_a is None or smi_b is None:
            break

        img_a = render_structure(smi_a, struct_size, cfg)
        img_b = render_structure(smi_b, struct_size, cfg)
        if img_a is None or img_b is None:
            continue

        # If label goes above, shift structure down by id_h
        struct_top = y_cursor + struct_pad + (id_h if label_above else 0)

        for idx, (smi, img) in enumerate([(smi_a, img_a), (smi_b, img_b)]):
            sw, sh = img.size
            panel_x = margin + idx * (struct_panel_w + arrow_panel_w)
            sx = panel_x + (struct_panel_w - sw) // 2
            sy = struct_top + (struct_size - sh) // 2
            page.paste(img, (sx, sy), img)
            struct_box = (sx, sy, sx + sw, sy + sh)

            label = random_label()
            label_cx = panel_x + struct_panel_w // 2
            if label_above:
                label_cy = y_cursor + struct_pad + id_h // 2
            else:
                label_cy = struct_top + struct_size + id_h // 2 + struct_pad // 2

            label_box = draw_rotated_text(page, label, (label_cx, label_cy), id_font, 0.0)
            panels.append({
                "struct_box": struct_box,
                "label_box":  label_box,
                "label_text": label,
                "smiles":     smi,
            })

        # Arrow (horizontal, in centre panel)
        arrow_x0 = margin + struct_panel_w + int(20 * scale)
        arrow_x1 = margin + struct_panel_w + arrow_panel_w - int(20 * scale)
        arrow_y = struct_top + struct_size // 2
        _draw_arrow(
            draw, arrow_x0, arrow_y, arrow_x1,
            color=(80, 80, 80),
            width=max(1, int(2 * scale)),
            head=max(8, int(14 * scale)),
        )

        # Transformation text above arrow
        transform = random.choice(_TRANSFORMATIONS)
        tw, th = _text_size(draw, transform, ann_font)
        draw.text(
            (arrow_x0 + (arrow_x1 - arrow_x0 - tw) // 2,
             arrow_y - th - int(8 * scale)),
            transform, font=ann_font, fill=(40, 40, 40),
        )

        # Δ-value annotations below arrow (decorative, not annotated)
        dy = arrow_y + int(12 * scale)
        for dname, dfn in zip(delta_names, delta_fns):
            sign = random.choice(["+", "-"])
            val_str = f"\u0394{dname.split()[0]}: {sign}{dfn()}"
            vw, vh = _text_size(draw, val_str, delta_font)
            draw.text(
                (arrow_x0 + (arrow_x1 - arrow_x0 - vw) // 2, dy),
                val_str, font=delta_font, fill=(60, 60, 140),
            )
            dy += vh + int(4 * scale)

        # Thin separator between pairs
        sep_y = y_cursor + pair_h - int(6 * scale)
        draw.line(
            [(margin, sep_y), (cfg.page_w - margin, sep_y)],
            fill=(220, 220, 220), width=1,
        )
        y_cursor += pair_h

    return page, panels
