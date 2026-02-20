#!/usr/bin/env python3
"""
Visualize YOLO bounding box labels overlaid on synthetic page images.

Reads images + corresponding .txt label files, draws boxes, saves to an
output directory for quick human verification.

Usage:
    python scripts/visualize_labels.py
    python scripts/visualize_labels.py --split val --n 50 --out data/viz
    python scripts/visualize_labels.py --split train --n 20 --out data/viz
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BOX_WIDTH = 6                 # Pixels — thick enough to see on 2480×3508
LABEL_COLOR = (255, 255, 255) # White text on label tag
CLASS_COLORS = {
    0: (0, 200, 0),    # Green — chemical_structure
    1: (0, 100, 255),  # Blue  — compound_label
}


def load_default_font(size: int = 28) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Parse a YOLO .txt file and return pixel-space boxes with class IDs."""
    boxes = []
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, cx, cy, bw, bh = map(float, parts)
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": int(class_id)})
    return boxes


def draw_boxes(img: Image.Image, boxes: list[dict], font: ImageFont.ImageFont) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        class_id = box.get("class_id", 0)
        color = CLASS_COLORS.get(class_id, (255, 0, 0))

        draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_WIDTH)

        # Small index tag at top-left corner of the box
        tag = str(i)
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0] + 6, tb[3] - tb[1] + 4
        tx, ty = x1, max(0, y1 - th)
        draw.rectangle([tx, ty, tx + tw, ty + th], fill=color)
        draw.text((tx + 3, ty + 2), tag, fill=LABEL_COLOR, font=font)

    return out


def visualize_split(
    data_dir: Path,
    split: str,
    out_dir: Path,
    n: int,
    seed: int,
) -> None:
    img_dir = data_dir / split / "images"
    lbl_dir = data_dir / split / "labels"

    if not img_dir.exists():
        print(f"Image directory not found: {img_dir}")
        return

    # Collect all image paths that have a matching label file
    img_paths = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and (lbl_dir / p.with_suffix(".txt").name).exists()
    )

    if not img_paths:
        print(f"No labeled images found in {img_dir}")
        return

    random.seed(seed)
    sample = random.sample(img_paths, min(n, len(img_paths)))
    sample.sort()

    split_out = out_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    font = load_default_font(size=28)

    ok = skipped = 0
    for img_path in sample:
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  [skip] {img_path.name}: {e}")
            skipped += 1
            continue

        boxes = parse_yolo_labels(lbl_path, img.width, img.height)
        annotated = draw_boxes(img, boxes, font)

        out_path = split_out / img_path.name
        annotated.save(str(out_path))
        ok += 1

    print(f"[{split}] saved {ok} images to {split_out}  (skipped {skipped})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize YOLO labels on synthetic pages")
    parser.add_argument("--data", type=Path, default=Path("data/generated"),
                        help="Root of the generated dataset (default: data/generated)")
    parser.add_argument("--split", choices=["train", "val", "both"], default="both",
                        help="Which split(s) to visualize (default: both)")
    parser.add_argument("--n", type=int, default=30,
                        help="Number of images to sample per split (default: 30)")
    parser.add_argument("--out", type=Path, default=Path("data/viz"),
                        help="Output directory (default: data/viz)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        visualize_split(args.data, split, args.out, args.n, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
