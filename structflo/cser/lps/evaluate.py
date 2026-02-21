"""Page-level pair accuracy evaluation: LearnedMatcher vs HungarianMatcher.

Entry point: ``sf-eval-lps``

Usage::

    sf-eval-lps --weights runs/lps/scorer_best.pt --data-dir data/generated/val

The script creates "detections" from ground-truth bounding boxes (perfect
localisation, no missed detections) so that it isolates the quality of the
*matching* step alone.  This is the right evaluation before running the full
inference pipeline on real images.

Page-level accuracy: a page is correct only if **every** (structure, label)
pair is correctly matched.  A single swap on a page counts as a failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from structflo.cser.pipeline.matcher import HungarianMatcher
from structflo.cser.pipeline.models import BBox, Detection

_PROJECT_ROOT = Path(__file__).parents[3]
_DEFAULT_VAL_DIR = _PROJECT_ROOT / "data" / "generated" / "val"

_TOL = 2.0  # pixel tolerance for bbox centroid comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _centroid(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _centroids_close(a: list[float], b: list[float], tol: float = _TOL) -> bool:
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    return abs(ax - bx) < tol and abs(ay - by) < tol


def _page_correct(pairs, gt_structs: list[list[float]], gt_labels: list[list[float]]) -> bool:
    """Return True if every pair in *pairs* matches the ground truth association."""
    if len(pairs) != len(gt_structs):
        return False
    for pair in pairs:
        s_bbox = pair.structure.bbox.as_list()
        l_bbox = pair.label.bbox.as_list()
        # Find which GT structure this matches
        match_idx = next(
            (i for i, gs in enumerate(gt_structs) if _centroids_close(s_bbox, gs)),
            None,
        )
        if match_idx is None:
            return False
        if not _centroids_close(l_bbox, gt_labels[match_idx]):
            return False
    return True


def _build_detections(valid_entries: list[dict]) -> list[Detection]:
    detections = []
    for entry in valid_entries:
        detections.append(
            Detection(bbox=BBox(*entry["struct_bbox"]), conf=1.0, class_id=0)
        )
        detections.append(
            Detection(bbox=BBox(*entry["label_bbox"]), conf=1.0, class_id=1)
        )
    return detections


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def evaluate(
    val_dir: Path,
    weights: Path | str,
    device: str = "cuda",
    geometry_only: bool = False,
    max_pages: int | None = None,
) -> dict:
    """Compare HungarianMatcher and LearnedMatcher on ground-truth bounding boxes.

    Args:
        val_dir:       Split directory containing ``ground_truth/`` and ``images/``.
        weights:       Path to a trained scorer checkpoint.
        device:        Torch device string.
        geometry_only: Use geometric features only (skip visual crops).
        max_pages:     If set, evaluate only the first *max_pages* pages (for
                       quick sanity checks).

    Returns:
        Dict with keys ``total_pages``, ``hungarian_accuracy``,
        ``learned_accuracy``, ``improvement``.
    """
    # Late import so the module can be imported without torch installed
    from structflo.cser.lps.matcher import LearnedMatcher

    gt_dir = val_dir / "ground_truth"
    img_dir = val_dir / "images"

    hungarian = HungarianMatcher()
    learned = LearnedMatcher(weights=weights, device=device, geometry_only=geometry_only)

    h_correct = 0
    l_correct = 0
    total = 0

    json_files = sorted(gt_dir.glob("*.json"))
    if max_pages is not None:
        json_files = json_files[:max_pages]

    for json_path in json_files:
        entries: list[dict] = json.loads(json_path.read_text())
        valid = [e for e in entries if e.get("label_bbox") is not None]

        # Skip pages with fewer than 2 labelled structures (trivially correct)
        if len(valid) < 2:
            continue

        gt_structs = [e["struct_bbox"] for e in valid]
        gt_labels = [e["label_bbox"] for e in valid]
        detections = _build_detections(valid)

        # Load image for LearnedMatcher (optional for geom-only but passed anyway)
        stem = json_path.stem
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = img_dir / f"{stem}.png"
        img_np = np.array(Image.open(img_path).convert("L")) if img_path.exists() else None

        h_pairs = hungarian.match(detections)
        l_pairs = learned.match(detections, image=img_np)

        h_correct += int(_page_correct(h_pairs, gt_structs, gt_labels))
        l_correct += int(_page_correct(l_pairs, gt_structs, gt_labels))
        total += 1

        if total % 500 == 0:
            print(
                f"  {total:>5} pages  "
                f"Hungarian: {h_correct/total:.2%}  "
                f"Learned: {l_correct/total:.2%}"
            )

    h_acc = h_correct / max(total, 1)
    l_acc = l_correct / max(total, 1)
    return {
        "total_pages": total,
        "hungarian_accuracy": h_acc,
        "learned_accuracy": l_acc,
        "improvement": l_acc - h_acc,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate LearnedMatcher vs HungarianMatcher on page-level pair accuracy"
    )
    p.add_argument(
        "--weights",
        type=Path,
        required=True,
        help="Path to trained scorer checkpoint (scorer_best.pt)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_VAL_DIR,
        help="Validation split directory (default: data/generated/val)",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--geometry-only",
        action="store_true",
        help="Use geometric features only (skip visual CNN crops)",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Limit evaluation to first N pages (quick sanity check)",
    )

    args = p.parse_args()

    print(f"[eval] val dir  : {args.data_dir}")
    print(f"[eval] weights  : {args.weights}")
    print(f"[eval] device   : {args.device}")
    print()

    results = evaluate(
        val_dir=args.data_dir,
        weights=args.weights,
        device=args.device,
        geometry_only=args.geometry_only,
        max_pages=args.max_pages,
    )

    print()
    print(f"Pages evaluated  : {results['total_pages']:,}")
    print(f"Hungarian acc    : {results['hungarian_accuracy']:.2%}")
    print(f"Learned acc      : {results['learned_accuracy']:.2%}")
    delta = results["improvement"]
    sign = "+" if delta >= 0 else ""
    print(f"Improvement      : {sign}{delta:.2%}")


if __name__ == "__main__":
    main()
