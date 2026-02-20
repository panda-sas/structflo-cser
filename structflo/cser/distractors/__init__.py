"""Distractor element generators assembled into a single module."""

import random
from typing import List, Optional, Tuple

from PIL import Image

from structflo.cser._geometry import try_place_box
from structflo.cser.data.distractor_images import _pick_distractor_image
from structflo.cser.distractors.charts import (
    _gen_bar_chart,
    _gen_line_plot,
    _gen_pie_chart,
    _gen_scatter_plot,
)
from structflo.cser.distractors.shapes import (
    _gen_geometric_shapes,
    _gen_gradient_block,
    _gen_noise_patch,
)

# Re-export text element functions for convenience
from structflo.cser.distractors.text_elements import (  # noqa: F401
    add_arrow,
    add_caption,
    add_equation_fragment,
    add_footnote,
    add_journal_header,
    add_multiline_text_block,
    add_page_number,
    add_panel_border,
    add_prose_block,
    add_rgroup_table,
    add_section_header,
    add_stray_text,
)
from structflo.cser.config import PageConfig

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
