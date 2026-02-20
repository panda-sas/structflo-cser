"""Text rendering utilities: fonts, labels, and rotated text compositing."""

import random
import string
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from struct_labels._geometry import boxes_intersect
from struct_labels.config import PageConfig


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


def load_font(font_paths: List[Path], size: int, prefer_bold: bool = False) -> ImageFont.ImageFont:
    """Pick a random font from *font_paths* at the given pixel *size*.

    If *prefer_bold* is True, only bold-named fonts are considered (if any).
    Falls back to DejaVuSans or Pillow's built-in default on failure.
    """
    paths = list(font_paths)
    if prefer_bold:
        # Match common bold-weight naming patterns across font families:
        #   *-Bold.ttf, *-Bd*.ttf, *-B.ttf (Ubuntu), *-Demi*.otf (URW),
        #   *-Heavy*, *-Black*, *-SemiBold*, *-ExtraBold*
        import re as _re
        _BOLD_PAT = _re.compile(
            r"(bold|[-_]bd[-_]?|[-_]b\.ttf|demi|heavy|black|semibold|extrabold)",
            _re.IGNORECASE,
        )
        bold_paths = [p for p in paths if _BOLD_PAT.search(p.name)]
        if bold_paths:
            paths = bold_paths
    random.shuffle(paths)
    for path in paths:
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
        If given, a filled rectangle of this colour is drawn behind the text.

    Returns
    -------
    (x0, y0, x1, y1) — tight bounding box of the rendered label on *base*,
    padded by `annotation_margin` pixels on each side.
    """
    draw = ImageDraw.Draw(base)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    annotation_margin = 7
    if abs(angle) < 0.5:
        canvas_padding = annotation_margin + 2
    else:
        diag = int(np.hypot(text_w, text_h))
        canvas_padding = (diag - min(text_w, text_h)) // 2 + annotation_margin + 4

    padded_w = text_w + 2 * canvas_padding
    padded_h = text_h + 2 * canvas_padding

    text_img = Image.new("RGBA", (padded_w, padded_h), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)

    tx = canvas_padding - text_bbox[0]
    ty = canvas_padding - text_bbox[1]

    glyph_x0 = canvas_padding
    glyph_y0 = canvas_padding
    glyph_x1 = canvas_padding + text_w
    glyph_y1 = canvas_padding + text_h

    if bg_color is not None:
        bg_pad = 4
        text_draw.rectangle(
            [glyph_x0 - bg_pad, glyph_y0 - bg_pad,
             glyph_x1 + bg_pad, glyph_y1 + bg_pad],
            fill=bg_color + (255,),
        )

    text_draw.text((tx, ty), text, font=font, fill=fill + (255,))

    rotated = text_img.rotate(angle, expand=True)

    cx, cy = position
    page_w, page_h = base.size
    x0 = cx - rotated.size[0] // 2
    y0 = cy - rotated.size[1] // 2
    x0 = max(0, min(x0, page_w - rotated.size[0]))
    y0 = max(0, min(y0, page_h - rotated.size[1]))

    base.paste(rotated, (x0, y0), rotated)

    alpha = np.array(rotated)[:, :, 3]
    ys, xs = np.where(alpha > 10)
    if len(ys) == 0:
        return (x0, y0, x0 + rotated.size[0], y0 + rotated.size[1])
    bx0 = max(0, xs.min() - annotation_margin)
    by0 = max(0, ys.min() - annotation_margin)
    bx1 = min(rotated.size[0], xs.max() + 1 + annotation_margin)
    by1 = min(rotated.size[1], ys.max() + 1 + annotation_margin)
    return (int(x0 + bx0), int(y0 + by0), int(x0 + bx1), int(y0 + by1))


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
    label = random_label()

    font_size = random.randint(*cfg.label_font_range)
    use_bold = random.random() < 0.5
    font = load_font(font_paths, font_size, prefer_bold=use_bold)

    x0, y0, x1, y1 = struct_box

    draw = ImageDraw.Draw(page)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    placement = random.choices(
        ["below", "above", "left", "right"],
        weights=[50, 20, 15, 15],
    )[0]
    offset = random.randint(*cfg.label_offset_range)

    struct_cx = x0 + (x1 - x0) // 2
    struct_cy = y0 + (y1 - y0) // 2

    if placement == "below":
        cx = struct_cx
        cy = y1 + offset + text_h // 2
        angle = 0.0
    elif placement == "above":
        cx = struct_cx
        cy = y0 - offset - text_h // 2
        angle = 0.0
    elif placement == "left":
        cx = x0 - offset - text_w // 2
        cy = struct_cy
        angle = 0.0
    else:  # right
        cx = x1 + offset + text_w // 2
        cy = struct_cy
        angle = 0.0

    if placement in ("left", "right") and random.random() < cfg.label_90deg_prob:
        angle = 90.0
    elif random.random() < cfg.label_rotation_prob:
        angle = random.uniform(*cfg.label_rotation_range)

    if random.random() < 0.30:
        bg_color = random.choice([
            (0, 0, 0),
            (30, 30, 30),
            (0, 0, 100),
            (80, 0, 0),
            (0, 60, 0),
            (60, 0, 80),
        ])
        fill_color = (255, 255, 255)
    else:
        bg_color = None
        fill_color = (0, 0, 0)

    label_box = draw_rotated_text(page, label, (cx, cy), font, angle,
                                  fill=fill_color, bg_color=bg_color)
    return label_box, label
