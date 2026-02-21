"""Neural network models for learned pair scoring.

Two variants:

* ``GeomScorer``  — MLP on geometric features only.  Fast, no image required.
  Start here; it is likely sufficient for the majority of pages.

* ``VisualScorer`` — Small CNN branches on image crops concatenated with
  geometric features.  Adds distractor rejection at the cost of requiring
  the page image at inference time.

Both output a **raw logit** (not sigmoid-normalised).  Use
``BCEWithLogitsLoss`` during training and ``torch.sigmoid`` at inference.

Checkpoint helpers ``save_checkpoint`` / ``load_checkpoint`` store the model
type alongside the state dict so that ``LearnedMatcher`` can reconstruct the
correct architecture from a checkpoint file without any extra metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from structflo.cser.lps.features import GEOM_DIM, LABEL_CROP_SIZE, STRUCT_CROP_SIZE


# ---------------------------------------------------------------------------
# Geometry-only scorer
# ---------------------------------------------------------------------------


class GeomScorer(nn.Module):
    """MLP association scorer using geometric features only.

    Input:  float32 tensor ``(..., GEOM_DIM)``
    Output: float32 tensor ``(..., 1)`` — raw logit
    """

    def __init__(self, input_dim: int = GEOM_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, geom: Tensor) -> Tensor:  # noqa: D102
        return self.net(geom)


# ---------------------------------------------------------------------------
# Visual scorer
# ---------------------------------------------------------------------------


class _SmallCNN(nn.Module):
    """Lightweight CNN: (B, 1, H, W) → (B, out_dim)."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(128, out_dim), nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.proj(self.body(x))


class VisualScorer(nn.Module):
    """Association scorer using image crops and geometric features.

    Inputs:
        struct_crop: ``(B, 1, H_s, W_s)`` float32 grayscale structure crop
        label_crop:  ``(B, 1, H_l, W_l)`` float32 grayscale label crop
        geom:        ``(B, GEOM_DIM)``     float32 geometric features

    Output: ``(B, 1)`` raw logit
    """

    STRUCT_FEAT = 64
    LABEL_FEAT = 32

    def __init__(self, geom_dim: int = GEOM_DIM) -> None:
        super().__init__()
        self.struct_cnn = _SmallCNN(self.STRUCT_FEAT)
        self.label_cnn = _SmallCNN(self.LABEL_FEAT)
        self.geom_proj = nn.Sequential(nn.Linear(geom_dim, 32), nn.ReLU())

        total = self.STRUCT_FEAT + self.LABEL_FEAT + 32
        self.head = nn.Sequential(
            nn.Linear(total, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, struct_crop: Tensor, label_crop: Tensor, geom: Tensor) -> Tensor:  # noqa: D102
        sf = self.struct_cnn(struct_crop)
        lf = self.label_cnn(label_crop)
        gf = self.geom_proj(geom)
        return self.head(torch.cat([sf, lf, gf], dim=-1))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_MODEL_TYPE_KEY = "model_type"
_STATE_DICT_KEY = "state_dict"


def save_checkpoint(model: nn.Module, path: Path, **meta: Any) -> None:
    """Save *model* and arbitrary *meta* fields to *path*.

    The ``model_type`` field is added automatically so that
    ``load_checkpoint`` can reconstruct the correct architecture.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        _MODEL_TYPE_KEY: "geom" if isinstance(model, GeomScorer) else "visual",
        _STATE_DICT_KEY: model.state_dict(),
        **meta,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path | str,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a checkpoint saved by ``save_checkpoint``.

    Returns:
        ``(model, meta)`` where *model* is a ``GeomScorer`` or
        ``VisualScorer`` with weights loaded, and *meta* is the rest of the
        checkpoint dict (epoch, val metrics, etc.).
    """
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    model_type = ckpt.get(_MODEL_TYPE_KEY, "geom")

    if model_type == "geom":
        model: nn.Module = GeomScorer()
    else:
        model = VisualScorer()

    model.load_state_dict(ckpt[_STATE_DICT_KEY])
    model.to(device)
    model.eval()

    meta = {k: v for k, v in ckpt.items() if k not in (_MODEL_TYPE_KEY, _STATE_DICT_KEY)}
    return model, meta
