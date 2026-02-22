"""Page generation configuration dataclass and factory functions."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PageConfig:
    # Page dimensions (A4 @ 300 DPI)
    page_w: int = 2480
    page_h: int = 3508
    margin: int = 180

    # Structure rendering
    struct_size_range: Tuple[int, int] = (280, 550)
    bond_width_range: Tuple[float, float] = (0.8, 5.0)   # was (1.5, 3.0); wider covers ChemDraw thin → bold scanned
    atom_font_range: Tuple[int, int] = (10, 36)           # was (14, 28)
    dark_bg_prob: float = 0.15                            # fraction of structures rendered on a dark background patch

    # Label rendering
    label_font_range: Tuple[int, int] = (8, 48)           # was (12, 36)
    label_offset_range: Tuple[int, int] = (10, 20)
    label_rotation_prob: float = 0.15
    label_rotation_range: Tuple[int, int] = (-15, 15)
    label_90deg_prob: float = 0.03

    # Page background
    page_bg_tint_prob: float = 0.12                       # cream / light-grey background instead of pure white

    # Layout
    min_structures: int = 1
    max_structures: int = 10
    two_column_prob: float = 0.25
    grid_layout_prob: float = 0.20  # 20% chance of uniform grid layout
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
        # Scale atom fonts with DPI so labels don't crowd structures at lower resolutions.
        atom_font_range=(max(7, int(10 * s)), max(12, int(36 * s))),
        label_font_range=(max(6, int(8 * s)), max(12, int(48 * s))),
        # Wider offset range (5→100 px at 300 DPI) breaks the tight dist_norm cluster
        # that a model can exploit when all synthetic labels are 10-20 px from structures.
        label_offset_range=(max(3, int(5 * s)), max(15, int(100 * s))),
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
        atom_font_range=(max(6, int(8 * s)), max(10, int(22 * s))),
        label_font_range=(max(8, int(10 * s)), max(12, int(26 * s))),
        label_offset_range=(max(2, int(4 * s)), max(10, int(50 * s))),
        min_structures=1,
        max_structures=6,  # slides pack fewer compounds
        two_column_prob=0.20,
        grid_layout_prob=0.35,  # grids are common in slide figures
    )
