"""Training script for the Learned Pair Scorer.

Entry point: ``sf-train-lps``

Usage (geometry-only, recommended first run)::

    sf-train-lps --data-dir data/generated --epochs 30

Usage (visual scorer)::

    sf-train-lps --data-dir data/generated --visual --epochs 30 --batch-size 512

The script trains a ``GeomScorer`` (default) or ``VisualScorer`` (``--visual``)
on positive and hard-negative pairs derived from the GT JSON files.  The best
checkpoint by validation accuracy is saved to ``runs/lps/``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from structflo.cser.lps.dataset import LPSDataset, PageGroupSampler
from structflo.cser.lps.scorer import GeomScorer, VisualScorer, save_checkpoint

_PROJECT_ROOT = Path(__file__).parents[3]
_DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "generated"
_DEFAULT_OUT_DIR = _PROJECT_ROOT / "runs" / "lps"


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    visual: bool,
    epoch: int,
) -> tuple[float, float]:
    """One training epoch.  Returns (mean_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    bar = tqdm(loader, desc=f"Epoch {epoch:>3} train", leave=False, unit="batch")
    for batch in bar:
        geom = batch["geom"].to(device)
        target = batch["target"].to(device).unsqueeze(1)

        if visual:
            sc = batch["struct_crop"].to(device)
            lc = batch["label_crop"].to(device)
            logits = model(sc, lc, geom)
        else:
            logits = model(geom)

        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = target.size(0)
        total_loss += loss.item() * bs
        preds = (logits.detach().sigmoid() >= 0.5).float()
        correct += (preds == target).sum().item()
        n += bs
        bar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{correct/n:.2%}")

    return total_loss / max(n, 1), correct / max(n, 1)


@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.BCEWithLogitsLoss,
    device: torch.device,
    visual: bool,
    epoch: int,
) -> tuple[float, float]:
    """One validation epoch.  Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0

    bar = tqdm(loader, desc=f"Epoch {epoch:>3}   val", leave=False, unit="batch")
    for batch in bar:
        geom = batch["geom"].to(device)
        target = batch["target"].to(device).unsqueeze(1)

        if visual:
            sc = batch["struct_crop"].to(device)
            lc = batch["label_crop"].to(device)
            logits = model(sc, lc, geom)
        else:
            logits = model(geom)

        loss = criterion(logits, target)
        bs = target.size(0)
        total_loss += loss.item() * bs
        preds = (logits.sigmoid() >= 0.5).float()
        correct += (preds == target).sum().item()
        n += bs
        bar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{correct/n:.2%}")

    return total_loss / max(n, 1), correct / max(n, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(
    data_dir: Path = _DEFAULT_DATA_DIR,
    output_dir: Path = _DEFAULT_OUT_DIR,
    epochs: int = 30,
    batch_size: int = 2048,
    visual: bool = False,
    neg_per_pos: int = 3,
    bbox_jitter: float = 0.02,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 8,
    device_str: str = "cuda",
    seed: int = 42,
) -> Path:
    """Train the LPS scorer and return the path to the best checkpoint."""
    torch.manual_seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lps] device      : {device}")
    print(f"[lps] data        : {data_dir}")
    print(f"[lps] output      : {output_dir}")
    print(f"[lps] scorer      : {'visual' if visual else 'geom-only'}")
    print(f"[lps] epochs      : {epochs}  batch: {batch_size}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print("[lps] building training dataset …")
    t0 = time.time()
    train_ds = LPSDataset(
        data_dir / "train",
        visual=visual,
        neg_per_pos=neg_per_pos,
        bbox_jitter=bbox_jitter,
        seed=seed,
    )
    print(f"[lps] train pairs : {len(train_ds):,}  ({time.time()-t0:.1f}s)")

    print("[lps] building validation dataset …")
    t0 = time.time()
    val_ds = LPSDataset(
        data_dir / "val",
        visual=visual,
        neg_per_pos=neg_per_pos,
        bbox_jitter=0.0,   # no jitter for validation
        seed=seed,
    )
    print(f"[lps] val pairs   : {len(val_ds):,}  ({time.time()-t0:.1f}s)")

    pw = train_ds.pos_weight()
    print(f"[lps] pos_weight  : {pw:.2f}")

    # spawn: avoids inheriting CUDA/libjpeg state (required for visual mode).
    # fork: faster startup, safe for geom-only (workers never open images/CUDA).
    # persistent_workers: keeps workers alive across epochs — avoids per-epoch
    #   respawn cost, especially important with spawn context.
    # prefetch_factor: each worker queues N batches ahead so the GPU is never
    #   idle waiting for data.
    _nw = num_workers
    _mp_ctx = "spawn" if visual else "fork"
    loader_kw: dict = dict(
        batch_size=batch_size,
        num_workers=_nw,
        pin_memory=(device.type == "cuda"),
        multiprocessing_context=_mp_ctx,
        persistent_workers=(_nw > 0),
    )
    if _nw > 0:
        loader_kw["prefetch_factor"] = 8

    if visual:
        # PageGroupSampler yields all samples from a page consecutively so the
        # per-worker LRU image cache gets ~20 hits per JPEG decode instead of 1.
        # This cuts I/O from O(N_samples) to O(N_pages) per epoch.
        train_sampler = PageGroupSampler(train_ds._path_idx, shuffle=True, seed=seed)
        val_sampler = PageGroupSampler(val_ds._path_idx, shuffle=False, seed=seed)
        train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kw)
        val_loader = DataLoader(val_ds, sampler=val_sampler, **loader_kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model: nn.Module = VisualScorer() if visual else GeomScorer()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lps] parameters  : {n_params:,}")

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pw], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_acc = 0.0
    best_path = output_dir / "scorer_best.pt"

    print(f"\n{'Epoch':>5}  {'TrainLoss':>9}  {'TrainAcc':>8}  {'ValLoss':>7}  {'ValAcc':>6}  {'LR':>8}")
    print("-" * 58)

    for epoch in range(1, epochs + 1):
        if visual:
            train_sampler.set_epoch(epoch)
        t_start = time.time()
        tr_loss, tr_acc = _train_epoch(model, train_loader, criterion, optimizer, device, visual, epoch)
        vl_loss, vl_acc = _val_epoch(model, val_loader, criterion, device, visual, epoch)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_start
        marker = " *" if vl_acc > best_acc else ""

        print(
            f"{epoch:>5}  {tr_loss:>9.4f}  {tr_acc:>7.2%}  "
            f"{vl_loss:>7.4f}  {vl_acc:>5.2%}  {lr_now:>8.2e}"
            f"  {elapsed:.0f}s{marker}"
        )

        if vl_acc > best_acc:
            best_acc = vl_acc
            save_checkpoint(
                model,
                best_path,
                epoch=epoch,
                val_accuracy=vl_acc,
                val_loss=vl_loss,
                visual=visual,
            )

    # Always save the final epoch checkpoint too
    final_path = output_dir / "scorer_final.pt"
    save_checkpoint(
        model,
        final_path,
        epoch=epochs,
        val_accuracy=vl_acc,
        val_loss=vl_loss,
        visual=visual,
    )

    print(f"\n[lps] best val accuracy : {best_acc:.2%}")
    print(f"[lps] best checkpoint   : {best_path}")
    print(f"[lps] final checkpoint  : {final_path}")
    return best_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train the Learned Pair Scorer (LPS) for structure-label association"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Root of generated data (must contain train/ and val/ subdirs)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory for checkpoints (default: runs/lps/)",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--batch",
        type=int,
        default=2048,
        help="Batch size (2048 is fast on A6000 for geom-only; use 512 for visual)",
    )
    p.add_argument(
        "--visual",
        action="store_true",
        help="Train VisualScorer (CNN + geom).  Default: GeomScorer (geom only)",
    )
    p.add_argument(
        "--neg-per-pos",
        type=int,
        default=3,
        help="Hard negatives generated per positive pair (default: 3)",
    )
    p.add_argument(
        "--bbox-jitter",
        type=float,
        default=0.02,
        help="Bbox coordinate jitter fraction to simulate YOLO noise (default: 0.02)",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch,
        visual=args.visual,
        neg_per_pos=args.neg_per_pos,
        bbox_jitter=args.bbox_jitter,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.workers,
        device_str=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
