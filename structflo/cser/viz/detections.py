"""Matplotlib-based visualisation for pipeline Detection / CompoundPair objects.

All public helpers accept ``PIL.Image``, a file path, or a NumPy array and
return a ``matplotlib.figure.Figure`` so the caller can save, show, or embed
the result in a notebook.

Typical usage::

    from structflo.cser.viz import plot_detections, plot_pairs, plot_results

    fig = plot_detections(image, detections)
    fig = plot_pairs(image, pairs)
    fig = plot_results(image, enriched_pairs)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from structflo.cser.pipeline.models import CompoundPair, Detection

# ── colour palette ──────────────────────────────────────────────────────────
CLASS_COLORS: dict[int, str] = {0: "lime", 1: "deepskyblue"}
CLASS_NAMES: dict[int, str] = {0: "structure", 1: "label"}
PAIR_COLOR: str = "orange"

# ── image coercion ──────────────────────────────────────────────────────────
ImageLike = Union[Path, str, np.ndarray, "Image.Image"]


def _to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


# ── internal helpers ────────────────────────────────────────────────────────


def _prepare_axes(
    image: ImageLike,
    ax: plt.Axes | None,
    figsize: tuple[float, float],
) -> tuple[Image.Image, Figure, plt.Axes]:
    """Return *(pil_image, figure, axes)*, creating a new figure when needed."""
    pil = _to_pil(image)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.get_figure()
    ax.imshow(pil)
    ax.axis("off")
    return pil, fig, ax


def _draw_detection_box(ax: plt.Axes, det: Detection) -> None:
    """Draw a single detection bounding box on *ax*."""
    b = det.bbox
    color = CLASS_COLORS.get(det.class_id, "red")
    ax.add_patch(
        mpatches.Rectangle(
            (b.x1, b.y1),
            b.width,
            b.height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
    )
    ax.text(
        b.x1,
        b.y1 - 4,
        f"{CLASS_NAMES.get(det.class_id, str(det.class_id))} {det.conf:.2f}",
        color=color,
        fontsize=8,
        weight="bold",
        bbox=dict(facecolor="black", alpha=0.5, pad=1),
    )


def _draw_pair_line(ax: plt.Axes, pair: CompoundPair, index: int) -> None:
    """Draw an orange centroid-to-centroid line for a matched pair."""
    sc = pair.structure.bbox.centroid
    lc = pair.label.bbox.centroid
    ax.plot([sc[0], lc[0]], [sc[1], lc[1]], color=PAIR_COLOR, linewidth=2)
    mid_x = (sc[0] + lc[0]) / 2
    mid_y = (sc[1] + lc[1]) / 2
    ax.text(mid_x + 5, mid_y, str(index), color=PAIR_COLOR, fontsize=12, weight="bold")


# ── public API ──────────────────────────────────────────────────────────────


def plot_detections(
    image: ImageLike,
    detections: Sequence[Detection],
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 14),
    title: str | None = None,
) -> Figure:
    """Overlay coloured bounding boxes on *image*.

    Parameters
    ----------
    image:
        Input image (path, PIL, or ndarray).
    detections:
        ``Detection`` objects returned by :pymethod:`ChemPipeline.detect`.
    ax:
        Existing matplotlib axes to draw on.  A new figure is created when
        *None* (default).
    figsize:
        Figure size when creating a new figure.
    title:
        Optional title.  Auto-generated when *None*.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _, fig, ax = _prepare_axes(image, ax, figsize)

    for det in detections:
        _draw_detection_box(ax, det)

    if title is None:
        n_s = sum(1 for d in detections if d.class_id == 0)
        n_l = sum(1 for d in detections if d.class_id == 1)
        title = f"Detections: {n_s} structures (green), {n_l} labels (blue)"
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pairs(
    image: ImageLike,
    pairs: Sequence[CompoundPair],
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 14),
    title: str | None = None,
) -> Figure:
    """Overlay matched pairs with bounding boxes and orange pairing lines.

    Parameters
    ----------
    image:
        Input image (path, PIL, or ndarray).
    pairs:
        ``CompoundPair`` objects returned by :pymethod:`ChemPipeline.match`.
    ax:
        Existing matplotlib axes to draw on.
    figsize:
        Figure size when creating a new figure.
    title:
        Optional title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _, fig, ax = _prepare_axes(image, ax, figsize)

    for i, pair in enumerate(pairs):
        _draw_detection_box(ax, pair.structure)
        _draw_detection_box(ax, pair.label)
        _draw_pair_line(ax, pair, i)

    ax.set_title(title or f"{len(pairs)} matched pairs")
    fig.tight_layout()
    return fig


def plot_crops(
    image: ImageLike,
    pairs: Sequence[CompoundPair],
    *,
    figsize_width: float = 8,
    row_height: float = 3,
    title: str | None = None,
) -> Figure:
    """Show each pair as a side-by-side (structure, label) crop gallery.

    Parameters
    ----------
    image:
        Input image to crop from.
    pairs:
        ``CompoundPair`` objects.
    figsize_width:
        Width of the figure in inches.
    row_height:
        Height of each row in inches.
    title:
        Optional suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pil = _to_pil(image)
    n = len(pairs)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(figsize_width, 2))
        ax.text(0.5, 0.5, "No pairs to display", ha="center", va="center")
        ax.axis("off")
        return fig

    fig, axes = plt.subplots(n, 2, figsize=(figsize_width, row_height * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, pair in enumerate(pairs):
        # Structure crop
        sb = pair.structure.bbox
        struct_crop = pil.crop((int(sb.x1), int(sb.y1), int(sb.x2), int(sb.y2)))
        axes[i, 0].imshow(struct_crop)
        axes[i, 0].set_title(
            f"Pair {i} — Structure (conf={pair.structure.conf:.2f})", fontsize=10
        )
        axes[i, 0].axis("off")

        # Label crop
        lb = pair.label.bbox
        label_crop = pil.crop((int(lb.x1), int(lb.y1), int(lb.x2), int(lb.y2)))
        axes[i, 1].imshow(label_crop)
        axes[i, 1].set_title(
            f"Pair {i} — Label (conf={pair.label.conf:.2f})", fontsize=10
        )
        axes[i, 1].axis("off")

    fig.suptitle(title or "Cropped structure & label regions", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_results(
    image: ImageLike,
    pairs: Sequence[CompoundPair],
    *,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (10, 14),
    max_smiles_len: int = 40,
    title: str | None = None,
) -> Figure:
    """Full annotated image: boxes, pairing lines, and extracted SMILES / label text.

    Best used after :pymethod:`ChemPipeline.enrich` so ``pair.smiles`` and
    ``pair.label_text`` are populated.

    Parameters
    ----------
    image:
        Input image.
    pairs:
        Enriched ``CompoundPair`` objects.
    ax:
        Existing matplotlib axes.
    figsize:
        Figure size when creating a new figure.
    max_smiles_len:
        Truncate SMILES strings longer than this.
    title:
        Optional title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _, fig, ax = _prepare_axes(image, ax, figsize)

    for i, pair in enumerate(pairs):
        # Structure box (green)
        _draw_detection_box(ax, pair.structure)
        # Label box (blue)
        _draw_detection_box(ax, pair.label)
        # Pairing line
        _draw_pair_line(ax, pair, i)

        # Annotation with extracted info
        smiles = pair.smiles or "—"
        if len(smiles) > max_smiles_len:
            smiles = smiles[:max_smiles_len] + "…"
        label_txt = pair.label_text or "—"
        annotation = f"#{i}  {label_txt}\n{smiles}"
        sb = pair.structure.bbox
        ax.text(
            sb.x1,
            sb.y1 - 8,
            annotation,
            color="white",
            fontsize=7,
            weight="bold",
            bbox=dict(facecolor="black", alpha=0.7, pad=2),
            verticalalignment="bottom",
        )

    ax.set_title(
        title or "Full pipeline results — structures + labels + SMILES", fontsize=13
    )
    fig.tight_layout()
    return fig
