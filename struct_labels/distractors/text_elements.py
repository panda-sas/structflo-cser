"""Text-based distractor elements: prose, captions, footnotes, arrows, tables, etc."""

import random
import textwrap
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw

from struct_labels._geometry import boxes_intersect, try_place_box
from struct_labels.config import PageConfig
from struct_labels.rendering.text import load_font

# ---------------------------------------------------------------------------
# Text corpora
# ---------------------------------------------------------------------------

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
    # Hard negatives: ID-like strings in non-label context
    "See CHEMBL4051 for details", "ZINC00123456 (inactive)",
    "Ref: PUBCHEM2341157", "cf. MCULE-1234567",
    "CPD-00441 showed no activity", "data from MOL-98231",
    "activity vs. ENAMINE-T001", "HIT-00923 excluded",
    "IC50 values: STD-00001–STD-00010",
    "Selected from ZINC library", "PUBCHEM CID: 44259",
]

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

# ---------------------------------------------------------------------------
# Distractor drawing functions
# ---------------------------------------------------------------------------


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
        draw.line((x0 + i * cell_w, y0, x0 + i * cell_w, y0 + rows * cell_h),
                  fill=(0, 0, 0), width=1)
    for j in range(rows + 1):
        draw.line((x0, y0 + j * cell_h, x0 + cols * cell_w, y0 + j * cell_h),
                  fill=(0, 0, 0), width=1)

    font = load_font(font_paths, 14)
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
            if (r == 0 or c == 0) and random.random() < 0.25:
                txt = random.choice(id_choices)
            else:
                txt = random.choice(cell_choices)
            draw.text((x0 + c * cell_w + 6, y0 + r * cell_h + 6), txt,
                      font=font, fill=(0, 0, 0))


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
