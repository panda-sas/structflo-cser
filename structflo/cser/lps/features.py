"""Geometric and visual feature extraction for (structure, label) candidate pairs.

Pure functions with no PyTorch or pipeline dependencies.  Called by
``dataset.py`` during training and by ``matcher.py`` at inference time.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEOM_DIM = 14  # length of the geometric feature vector produced by geom_features()

# Fixed crop sizes fed to the CNN branches of VisualScorer.
# (H, W) convention — structures are square, labels are wide/flat for text.
STRUCT_CROP_SIZE: tuple[int, int] = (128, 128)
LABEL_CROP_SIZE: tuple[int, int] = (32, 96)


# ---------------------------------------------------------------------------
# Geometric features
# ---------------------------------------------------------------------------


def geom_features(
    struct_bbox: Sequence[float],  # [x1, y1, x2, y2] pixels
    label_bbox: Sequence[float],   # [x1, y1, x2, y2] pixels
    page_w: float,
    page_h: float,
    struct_conf: float = 1.0,
    label_conf: float = 1.0,
) -> np.ndarray:
    """Return a float32 vector of ``GEOM_DIM`` scale-invariant features.

    All spatial values are normalised so the vector is resolution-independent.
    A model trained on synthetic A4@300 DPI pages generalises to real-world
    pages at different resolutions without any fine-tuning.

    Feature layout (14 values):
        0  dx_norm       lateral offset (label_cx - struct_cx) / struct_w
        1  dy_norm       vertical offset (label_cy - struct_cy) / struct_h
        2  dist_norm     Euclidean distance in normalised space
        3  angle_sin     sin of atan2(dy_norm, dx_norm)  — avoids discontinuity
        4  angle_cos     cos of atan2(dy_norm, dx_norm)
        5  size_ratio    label_area / struct_area
        6  label_aspect  label_w / label_h
        7  struct_aspect struct_w / struct_h
        8  struct_page_x struct centroid x / page_w
        9  struct_page_y struct centroid y / page_h
       10  label_page_x  label centroid x / page_w
       11  label_page_y  label centroid y / page_h
       12  struct_conf   YOLO detection confidence (1.0 for GT boxes)
       13  label_conf    YOLO detection confidence (1.0 for GT boxes)
    """
    sx1, sy1, sx2, sy2 = struct_bbox
    lx1, ly1, lx2, ly2 = label_bbox

    scx, scy = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0
    lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0

    sw = max(sx2 - sx1, 1e-6)
    sh = max(sy2 - sy1, 1e-6)
    lw = max(lx2 - lx1, 1e-6)
    lh = max(ly2 - ly1, 1e-6)

    dx_norm = (lcx - scx) / sw
    dy_norm = (lcy - scy) / sh
    dist_norm = math.hypot(dx_norm, dy_norm)

    angle = math.atan2(dy_norm, dx_norm)

    return np.array(
        [
            dx_norm,
            dy_norm,
            dist_norm,
            math.sin(angle),
            math.cos(angle),
            (lw * lh) / max(sw * sh, 1e-6),  # size_ratio
            lw / lh,                           # label_aspect
            sw / sh,                           # struct_aspect
            scx / max(page_w, 1.0),            # struct_page_x
            scy / max(page_h, 1.0),            # struct_page_y
            lcx / max(page_w, 1.0),            # label_page_x
            lcy / max(page_h, 1.0),            # label_page_y
            float(struct_conf),
            float(label_conf),
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Visual crops
# ---------------------------------------------------------------------------


def crop_region(
    image: np.ndarray,
    bbox: Sequence[float],
    out_size: tuple[int, int],
    jitter: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Crop *bbox* from *image*, resize to *out_size*, return float32 (1, H, W).

    Args:
        image:    Source image as uint8 numpy array, shape (H, W) or (H, W, 3).
        bbox:     Bounding box [x1, y1, x2, y2] in pixel coordinates.
        out_size: Target ``(height, width)`` of the output crop.
        jitter:   If > 0, apply uniform noise ±jitter×box_side to each
                  coordinate before cropping.  Simulates YOLO localisation
                  noise during training.  Requires *rng* to be set.
        rng:      NumPy Generator used for jitter (required when jitter > 0).

    Returns:
        float32 array of shape ``(1, H, W)`` in [0, 1] — channel-first
        single-channel grayscale, ready to be used as a PyTorch input.
    """
    from PIL import Image  # local import: keeps numpy-only callers free of PIL

    x1, y1, x2, y2 = (float(v) for v in bbox)

    if jitter > 0.0 and rng is not None:
        bw, bh = x2 - x1, y2 - y1
        noise = rng.uniform(-jitter, jitter, 4) * np.array([bw, bh, bw, bh])
        x1, y1, x2, y2 = x1 + noise[0], y1 + noise[1], x2 + noise[2], y2 + noise[3]

    img_h, img_w = image.shape[:2]
    x1i = max(0, int(round(x1)))
    y1i = max(0, int(round(y1)))
    x2i = min(img_w, max(x1i + 1, int(round(x2))))
    y2i = min(img_h, max(y1i + 1, int(round(y2))))

    patch = image[y1i:y2i, x1i:x2i]
    pil = Image.fromarray(patch).convert("L") if patch.ndim == 3 else Image.fromarray(patch, mode="L")

    out_h, out_w = out_size
    pil = pil.resize((out_w, out_h), Image.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0  # (H, W)
    return arr[np.newaxis]  # (1, H, W)
