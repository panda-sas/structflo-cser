#!/usr/bin/env python3
"""
Fine-tune YOLO11l for compound panel detection.

Detection target: union bbox of chemical structure + label ID (1 class).
Hardware: NVIDIA A6000 Ada (48 GB VRAM)

Usage:
    python scripts/03_train.py                        # defaults
    python scripts/03_train.py --weights yolo11m.pt   # lighter model
    python scripts/03_train.py --resume runs/...      # resume from checkpoint
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
DATA_YAML = ROOT / "config" / "data.yaml"
RUNS_DIR = ROOT / "runs" / "labels_detect"


def train(
    weights: str = "yolo11l.pt",
    imgsz: int = 1280,
    batch: int = 16,
    epochs: int = 150,
    resume: str | None = None,
) -> None:
    model = YOLO(resume if resume else weights)

    model.train(
        data=str(DATA_YAML),

        epochs=epochs,
        patience=30,
        batch=batch,
        imgsz=imgsz,

        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=5e-4,
        warmup_epochs=5,
        cos_lr=True,

        # Document augmentation — conservative, no spatial nonsense
        hsv_h=0.005,      # tiny hue shift (docs are mostly greyscale)
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=3.0,      # slight rotation (scanned page tilt)
        translate=0.1,
        scale=0.3,        # important: bridges train (1280) vs inference (1536 tile) scale gap
        shear=1.0,
        flipud=0.0,       # never: documents don't appear upside down
        fliplr=0.0,       # never: chemical handedness matters
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,

        cache="disk",     # preprocess once, reuse across epochs (~2000 large JPEGs)
        workers=8,

        project=str(RUNS_DIR),
        name="yolo11l_panels",
        exist_ok=False,
        seed=42,
        plots=True,
        save_period=25,   # checkpoint every 25 epochs as backup

        resume=bool(resume),
    )

    # Final val metrics on best weights
    best = RUNS_DIR / "yolo11l_panels" / "weights" / "best.pt"
    if best.exists():
        m = YOLO(str(best)).val(data=str(DATA_YAML))
        print(f"\n--- Best model metrics ---")
        print(f"mAP50:     {m.box.map50:.4f}   (target > 0.95)")
        print(f"mAP50-95:  {m.box.map:.4f}")
        print(f"Precision: {m.box.mp:.4f}")
        print(f"Recall:    {m.box.mr:.4f}")
        print(f"\nWeights: {best}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolo11l.pt",
                   help="Pretrained weights (default: yolo11l.pt)")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Training image size. 1280 is fine — compound panels are 200-350px "
                        "at this scale, well above detection threshold.")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size. 16 is safe for A6000 48GB at imgsz=1280.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--resume", default=None,
                   help="Path to last.pt to resume an interrupted run.")
    args = p.parse_args()

    if args.imgsz > 1280 and args.batch > 8:
        print(f"[warn] imgsz={args.imgsz} with batch={args.batch} may OOM. "
              f"Consider --batch 8.")

    train(args.weights, args.imgsz, args.batch, args.epochs, args.resume)


if __name__ == "__main__":
    main()
