"""Neural network model for learned pair scoring.

``PairScorer`` — the single model used at both training and inference.

Takes a structure image crop, a label image crop, and a geometric feature
vector; returns a raw logit (positive = likely true pair).

Checkpoint helpers ``save_checkpoint`` / ``load_checkpoint`` store the
state dict alongside training metadata (epoch, val metrics, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from structflo.cser.lps.features import GEOM_DIM


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Recalibrates channel responses by learning which feature maps are
    most informative for a given spatial input.
    """

    def __init__(self, ch: int, reduction: int = 8) -> None:
        super().__init__()
        bottleneck = max(ch // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, bottleneck, bias=False),
            nn.GELU(),
            nn.Linear(bottleneck, ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class _ResBlock(nn.Module):
    """2-conv residual block with SE channel attention and GELU activations."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.se = _SEBlock(ch)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.se(self.body(x)))


class _EmbedCNN(nn.Module):
    """CNN encoder: (B, 1, H, W) → (B, out_dim).

    Architecture::

        Stem:    Conv(1→32, 5×5) → BN → GELU → MaxPool(/2)
        Stage 1: ResBlock(32) → DownConv(32→64, stride=2)
        Stage 2: ResBlock(64) → DownConv(64→128, stride=2)
        Pool:    GlobalAvgPool ‖ GlobalMaxPool → 256-d
        Proj:    Linear(256→out_dim) → LayerNorm → GELU

    Spatial flow for struct crop (128×128):
        stem → 64×64  →  stage1 → 32×32  →  stage2 → 16×16  →  pool

    Spatial flow for label crop (64×96):
        stem → 32×48  →  stage1 → 16×24  →  stage2 →  8×12  →  pool

    Design rationale:
    - 5×5 stem captures rotated glyphs better than 3×3 (larger initial RF).
    - Residual connections stabilise gradient flow through 6+ layers.
    - SE attention re-weights channels per spatial input (distractor rejection).
    - Concatenated avg+max pool: avg = mean intensity, max = strongest
      local activation — together richer than avg alone for sparse text.
    """

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage1 = nn.Sequential(
            _ResBlock(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            _ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        avg = self.gap(x).flatten(1)
        mx = self.gmp(x).flatten(1)
        return self.proj(torch.cat([avg, mx], dim=1))


# ---------------------------------------------------------------------------
# Pair scorer
# ---------------------------------------------------------------------------


class PairScorer(nn.Module):
    """Structure-label association scorer.

    Inputs:
        struct_crop: ``(B, 1, 128, 128)`` float32 grayscale crop
        label_crop:  ``(B, 1,  64,  96)`` float32 grayscale crop
        geom:        ``(B, GEOM_DIM)``    float32 geometric features

    Output: ``(B, 1)`` raw logit — apply sigmoid for probability.

    Feature dimensions:
        struct embedding : 128-d
        label  embedding : 128-d
        geom projection  :  64-d
        concatenated     : 320-d → head → 1

    Parameter count: ~1.16 M
    """

    STRUCT_FEAT = 128
    LABEL_FEAT = 128
    GEOM_PROJ = 64

    def __init__(self, geom_dim: int = GEOM_DIM) -> None:
        super().__init__()
        self.struct_cnn = _EmbedCNN(self.STRUCT_FEAT)
        self.label_cnn = _EmbedCNN(self.LABEL_FEAT)
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, self.GEOM_PROJ),
            nn.LayerNorm(self.GEOM_PROJ),
            nn.GELU(),
        )
        total = self.STRUCT_FEAT + self.LABEL_FEAT + self.GEOM_PROJ
        self.head = nn.Sequential(
            nn.Linear(total, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, struct_crop: Tensor, label_crop: Tensor, geom: Tensor) -> Tensor:
        sf = self.struct_cnn(struct_crop)
        lf = self.label_cnn(label_crop)
        gf = self.geom_proj(geom)
        return self.head(torch.cat([sf, lf, gf], dim=-1))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_STATE_DICT_KEY = "state_dict"


def save_checkpoint(model: nn.Module, path: Path, **meta: Any) -> None:
    """Save *model* weights and arbitrary *meta* fields to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({_STATE_DICT_KEY: model.state_dict(), **meta}, path)


def load_checkpoint(
    path: Path | str,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a ``PairScorer`` checkpoint saved by ``save_checkpoint``.

    Returns:
        ``(model, meta)`` where *model* has weights loaded and is in eval
        mode, and *meta* is everything else in the checkpoint (epoch, val
        metrics, etc.).
    """
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    model = PairScorer()
    model.load_state_dict(ckpt[_STATE_DICT_KEY])
    model.to(device)
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != _STATE_DICT_KEY}
    return model, meta
