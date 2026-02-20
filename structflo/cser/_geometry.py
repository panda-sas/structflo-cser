"""Pure bounding-box geometry utilities shared across modules."""

import random
from typing import List, Optional, Tuple


def clamp_box(
    box: Tuple[int, int, int, int], w: int, h: int
) -> Tuple[int, int, int, int]:
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
        padded = (
            x0 - padding,
            y0 - padding,
            x0 + box_w + padding,
            y0 + box_h + padding,
        )
        if not any(boxes_intersect(padded, b) for b in existing):
            return box
    return None
