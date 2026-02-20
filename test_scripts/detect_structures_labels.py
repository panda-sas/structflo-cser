#!/usr/bin/env python3
"""
Detection test script — YOLO 2-class detector (chemical_structure + compound_label).

Runs the detector on a page image (or directory), draws coloured bounding
boxes (green=structure, blue=label), and saves the visualisation.

Usage:
    uv run python test_scripts/detect_structures_labels.py --image page.png
    uv run python test_scripts/detect_structures_labels.py --image_dir data/real/images/ --out detections/
    uv run python test_scripts/detect_structures_labels.py --image page.png --no_tile
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "labels_detect" / "yolo11l_panels" / "weights" / "last.pt"

CLASS_NAMES  = {0: "structure", 1: "label"}
CLASS_COLORS = {0: (0, 200, 0), 1: (0, 100, 255)}  # green, blue

# ── Tiling ────────────────────────────────────────────────────────────────────

def generate_tiles(img_w: int, img_h: int,
                   tile_size: int = 1536, overlap: float = 0.20) -> list[tuple]:
    step = int(tile_size * (1 - overlap))
    tiles = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            x_end = min(x + tile_size, img_w)
            y_end = min(y + tile_size, img_h)
            x_start = max(0, x_end - tile_size)
            y_start = max(0, y_end - tile_size)
            tiles.append((x_start, y_start, x_end, y_end))
            if x_end >= img_w:
                break
            x += step
        if y_end >= img_h:
            break
        y += step
    return tiles


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return np.array(keep)


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_tiled(model: YOLO, img: np.ndarray,
                 tile_size: int = 1536, overlap: float = 0.20,
                 conf: float = 0.3, nms_iou: float = 0.5) -> list[dict]:
    h, w = img.shape[:2]
    tiles = generate_tiles(w, h, tile_size, overlap)
    all_boxes, all_scores, all_classes = [], [], []

    for (x1, y1, x2, y2) in tiles:
        tile = img[y1:y2, x1:x2]
        results = model(tile, imgsz=tile_size, conf=conf, verbose=False)[0]
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
            all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
            all_scores.append(float(box.conf[0]))
            all_classes.append(int(box.cls[0]))

    if not all_boxes:
        return []

    boxes_arr  = np.array(all_boxes)
    scores_arr = np.array(all_scores)
    classes_arr = np.array(all_classes)

    # NMS per class so structure and label boxes don't suppress each other
    keep = []
    for cls_id in np.unique(classes_arr):
        mask = np.where(classes_arr == cls_id)[0]
        kept = nms(boxes_arr[mask], scores_arr[mask], nms_iou)
        keep.extend(mask[kept].tolist())

    return [{"bbox": boxes_arr[i].tolist(), "conf": float(scores_arr[i]),
             "class_id": int(classes_arr[i])} for i in keep]


def detect_full(model: YOLO, img: np.ndarray,
                conf: float = 0.3) -> list[dict]:
    results = model(img, conf=conf, verbose=False)[0]
    out = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        out.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(box.conf[0]),
                    "class_id": int(box.cls[0])})
    return out


# ── Visualisation ─────────────────────────────────────────────────────────────

def draw_boxes(img_pil: Image.Image, detections: list[dict]) -> Image.Image:
    vis = img_pil.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        cls = d.get("class_id", 0)
        color = CLASS_COLORS.get(cls, (255, 0, 0))
        name  = CLASS_NAMES.get(cls, str(cls))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 4, y1 + 4), f"{name} {d['conf']:.2f}", fill=color, font=font)

    return vis


# ── Main ──────────────────────────────────────────────────────────────────────

def process_image(model: YOLO, image_path: Path, out_dir: Path,
                  tile: bool, tile_size: int, conf: float,
                  rescale_dpi: int = 0, grayscale: bool = False) -> list[dict]:
    img_pil = Image.open(image_path).convert("RGB")
    dpi_info = img_pil.info.get("dpi", (None, None))
    src_dpi  = dpi_info[0] if dpi_info[0] else None
    print(f"  Image size: {img_pil.width}×{img_pil.height}  "
          f"DPI: {src_dpi if src_dpi else 'not set'}")

    if grayscale:
        img_pil = img_pil.convert("L").convert("RGB")

    scale = 1.0
    if rescale_dpi and src_dpi and abs(src_dpi - rescale_dpi) > 1:
        scale = rescale_dpi / src_dpi
        new_w = int(img_pil.width  * scale)
        new_h = int(img_pil.height * scale)
        print(f"  Rescaling {src_dpi:.0f} → {rescale_dpi} DPI  "
              f"({img_pil.width}×{img_pil.height} → {new_w}×{new_h})")
        img_pil = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img_np = np.array(img_pil)

    if tile:
        detections = detect_tiled(model, img_np, tile_size=tile_size, conf=conf)
    else:
        detections = detect_full(model, img_np, conf=conf)

    # Map boxes back to original pixel space for the saved visualisation
    orig_pil = Image.open(image_path).convert("RGB")
    if scale != 1.0:
        for d in detections:
            d["bbox"] = [v / scale for v in d["bbox"]]

    vis = draw_boxes(orig_pil, detections)
    out_path = out_dir / f"{image_path.stem}_detect.jpg"
    vis.save(str(out_path), quality=90)

    n_struct = sum(1 for d in detections if d.get("class_id", 0) == 0)
    n_label  = sum(1 for d in detections if d.get("class_id", 0) == 1)
    print(f"{image_path.name}: {n_struct} structure(s), {n_label} label(s) → {out_path.name}")
    for i, d in enumerate(detections):
        bb  = d["bbox"]
        cls = CLASS_NAMES.get(d.get("class_id", 0), "?")
        print(f"  [{i}] {cls:9s} conf={d['conf']:.3f}  "
              f"bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f})")
    return detections


def main():
    p = argparse.ArgumentParser(description="YOLO compound panel detection test")
    p.add_argument("--image",     help="Single image or PDF page (PNG/JPG)")
    p.add_argument("--image_dir", help="Directory of images")
    p.add_argument("--weights",   default=str(DEFAULT_WEIGHTS))
    p.add_argument("--out",       default="detections",
                   help="Output directory for visualisations")
    p.add_argument("--conf",      type=float, default=0.3,
                   help="Detection confidence threshold")
    p.add_argument("--tile_size", type=int,   default=1536)
    p.add_argument("--no_tile",     action="store_true",
                   help="Run on full image instead of tiling")
    p.add_argument("--rescale_dpi", type=int, default=0,
                   help="Rescale image to this DPI before detection (0 to disable). "
                        "Only useful if src DPI metadata is embedded.")
    p.add_argument("--grayscale", action="store_true",
                   help="Convert image to grayscale before detection (match grayscale training data).")
    args = p.parse_args()

    if not args.image and not args.image_dir:
        p.error("Provide --image or --image_dir")

    weights = Path(args.weights)
    if not weights.exists():
        p.error(f"Weights not found: {weights}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    print(f"Loaded weights: {weights}")
    print(f"Tiling: {'disabled' if args.no_tile else f'tile_size={args.tile_size}, overlap=20%'}")
    print(f"Conf threshold: {args.conf}  |  Grayscale: {args.grayscale}\n")

    if args.image:
        paths = [Path(args.image)]
    else:
        image_dir = Path(args.image_dir)
        paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))

    for path in paths:
        try:
            process_image(model, path, out_dir,
                          tile=not args.no_tile,
                          tile_size=args.tile_size,
                          conf=args.conf,
                          rescale_dpi=args.rescale_dpi,
                          grayscale=args.grayscale)
        except Exception as e:
            print(f"ERROR {path.name}: {e}")

    print(f"\nVisualisations saved to: {out_dir}/")


if __name__ == "__main__":
    main()
