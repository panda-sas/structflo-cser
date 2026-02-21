"""LearnedMatcher — BaseMatcher implementation backed by a trained scorer.

Replaces the Euclidean cost matrix in Hungarian matching with a learned
association probability.  The assignment algorithm itself (Hungarian /
``linear_sum_assignment``) is preserved; only the cost metric changes.

Usage::

    from structflo.cser.lps import LearnedMatcher
    from structflo.cser.pipeline import ChemPipeline

    pipeline = ChemPipeline(
        matcher=LearnedMatcher(weights="runs/lps/scorer_best.pt")
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from structflo.cser.lps.features import (
    LABEL_CROP_SIZE,
    STRUCT_CROP_SIZE,
    crop_region,
    geom_features,
)
from structflo.cser.lps.scorer import GeomScorer, VisualScorer, load_checkpoint
from structflo.cser.pipeline.matcher import BaseMatcher
from structflo.cser.pipeline.models import CompoundPair, Detection


class LearnedMatcher(BaseMatcher):
    """Pair structures with labels using a trained association scorer.

    The scorer produces a probability in [0, 1] for every candidate
    (structure, label) pair.  Hungarian matching is then run on
    ``(1 - score)`` as the cost matrix, preserving global optimality.

    ``match_distance`` on the returned ``CompoundPair`` objects is set to
    ``1 - score`` so that lower values mean higher confidence — consistent
    with the convention used by ``HungarianMatcher``.

    Args:
        weights:       Path to a scorer ``.pt`` checkpoint, a version tag
                       (``"v1.0"``), or ``None`` to auto-download the latest
                       published scorer from HuggingFace Hub.
        min_score:     Pairs whose association score is below this threshold
                       are dropped (structure considered unlabelled).
        device:        Torch device string (``"cuda"`` or ``"cpu"``).
        geometry_only: If ``True``, the visual CNN branches are skipped even
                       when the loaded model is a ``VisualScorer``.  Useful
                       when no image is available at inference time.
        max_dist_px:   Optional pre-filter: candidate pairs whose centroid
                       distance exceeds this value are excluded before
                       scoring, saving compute on dense pages.
    """

    def __init__(
        self,
        weights: Path | str | None = None,
        min_score: float = 0.5,
        device: str = "cuda",
        geometry_only: bool = False,
        max_dist_px: float | None = None,
    ) -> None:
        self.min_score = min_score
        self.geometry_only = geometry_only
        self.max_dist_px = max_dist_px
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = self._load(weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, weights: Path | str | None) -> torch.nn.Module:
        from structflo.cser.weights import resolve_weights

        path = resolve_weights("cser-lps", version=weights)
        model, _ = load_checkpoint(path, device=str(self._device))
        model.eval()
        return model

    def _page_size(
        self,
        detections: list[Detection],
        image: np.ndarray | None,
    ) -> tuple[float, float]:
        """Infer page dimensions from the image or from detection extents."""
        if image is not None:
            h, w = image.shape[:2]
            return float(w), float(h)
        # Fall back to detection bounding box extents (slightly underestimates)
        if detections:
            pw = max(d.bbox.x2 for d in detections)
            ph = max(d.bbox.y2 for d in detections)
            return float(pw), float(ph)
        return 2480.0, 3508.0  # A4 @ 300 DPI default

    def _score_matrix(
        self,
        structures: list[Detection],
        labels: list[Detection],
        image: np.ndarray | None,
        page_w: float,
        page_h: float,
    ) -> np.ndarray:
        """Compute the (n_struct × n_label) association score matrix."""
        n_s, n_l = len(structures), len(labels)
        visual = isinstance(self._model, VisualScorer) and not self.geometry_only

        if visual and image is None:
            raise ValueError(
                "LearnedMatcher loaded a VisualScorer but no image was provided. "
                "Pass image= to match(), or use geometry_only=True."
            )

        # Build flat lists of all n_s × n_l pairs
        geom_rows = []
        struct_crops = []
        label_crops = []

        for s in structures:
            for l in labels:
                geom_rows.append(
                    geom_features(
                        s.bbox.as_list(),
                        l.bbox.as_list(),
                        page_w,
                        page_h,
                        s.conf,
                        l.conf,
                    )
                )
                if visual:
                    struct_crops.append(crop_region(image, s.bbox.as_list(), STRUCT_CROP_SIZE))
                    label_crops.append(crop_region(image, l.bbox.as_list(), LABEL_CROP_SIZE))

        geom_t = torch.from_numpy(np.stack(geom_rows)).to(self._device)  # (n_s*n_l, GEOM_DIM)

        with torch.no_grad():
            if visual:
                sc_t = torch.from_numpy(np.stack(struct_crops)).to(self._device)
                lc_t = torch.from_numpy(np.stack(label_crops)).to(self._device)
                logits = self._model(sc_t, lc_t, geom_t)
            else:
                logits = self._model(geom_t)

            scores = logits.sigmoid().squeeze(1).cpu().numpy()  # (n_s*n_l,)

        return scores.reshape(n_s, n_l)

    # ------------------------------------------------------------------
    # BaseMatcher interface
    # ------------------------------------------------------------------

    def match(
        self,
        detections: list[Detection],
        image: np.ndarray | None = None,
    ) -> list[CompoundPair]:
        """Pair structures with labels using learned association scores.

        Args:
            detections: Flat list of all detections (structures + labels mixed).
            image:      Page image as a uint8 numpy array (H, W) or (H, W, 3).
                        Required for ``VisualScorer``; optional for ``GeomScorer``.

        Returns:
            List of ``CompoundPair`` objects.  ``match_distance`` is set to
            ``1 - score`` (lower = more confident match).
        """
        structures = [d for d in detections if d.class_id == 0]
        labels = [d for d in detections if d.class_id == 1]

        if not structures or not labels:
            return []

        page_w, page_h = self._page_size(detections, image)

        # Optional pre-filter: exclude candidate pairs that are too far apart
        # to avoid spending compute on obviously wrong assignments on dense pages.
        if self.max_dist_px is not None:
            valid_s, valid_l = set(), set()
            for i, s in enumerate(structures):
                scx, scy = s.bbox.centroid
                for j, l in enumerate(labels):
                    lcx, lcy = l.bbox.centroid
                    if ((scx - lcx) ** 2 + (scy - lcy) ** 2) ** 0.5 <= self.max_dist_px:
                        valid_s.add(i)
                        valid_l.add(j)
            structures = [structures[i] for i in sorted(valid_s)]
            labels = [labels[j] for j in sorted(valid_l)]
            if not structures or not labels:
                return []

        score_matrix = self._score_matrix(structures, labels, image, page_w, page_h)
        cost_matrix = 1.0 - score_matrix

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        pairs = []
        for r, c in zip(row_ind, col_ind):
            score = float(score_matrix[r, c])
            if score >= self.min_score:
                pairs.append(
                    CompoundPair(
                        structure=structures[r],
                        label=labels[c],
                        match_distance=float(1.0 - score),
                    )
                )
        return pairs
