"""Dataset generation: orchestrates page generation, saving, and multiprocessing."""

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from structflo.cser._geometry import clamp_box
from structflo.cser.config import make_page_config, make_page_config_slide
from structflo.cser.data.distractor_images import load_distractor_images
from structflo.cser.data.smiles import load_smiles
from structflo.cser.generation.page import apply_noise, make_negative_page, make_page
from structflo.cser.generation.specialty import make_data_card_page, make_mmp_page, make_sar_page
from structflo.cser.generation.tabular import make_excel_page, make_grid_page


def yolo_label(box: tuple, w: int, h: int, class_id: int = 0) -> str:
    """Convert a pixel bounding box to a YOLO-format annotation line.

    YOLO format: <class> <cx> <cy> <bw> <bh>  (all normalised 0-1).
    """
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0 / w
    cy = (y0 + y1) / 2.0 / h
    bw = (x1 - x0) / w
    bh = (y1 - y0) / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def find_fonts(fonts_dir: Optional[Path]) -> List[Path]:
    """Collect all .ttf and .otf font files from *fonts_dir* recursively."""
    if not fonts_dir or not fonts_dir.exists():
        return []
    font_paths = []
    for ext in ("*.ttf", "*.otf"):
        font_paths.extend(fonts_dir.rglob(ext))
    return font_paths


def save_sample(
    page,
    panels: List[dict],
    out_img: Path,
    out_lbl: Path,
    out_gt: Path,
    fmt: str,
    cfg,
    grayscale: bool = False,
) -> None:
    """Post-process and save one generated page.

    1. Apply random noise / JPEG artefacts / blur (data augmentation).
    2. Optionally convert to grayscale (keeps 3-channel for YOLO compat).
    3. Write the image, YOLO label file, and ground-truth JSON.
    """
    page = apply_noise(page, cfg)

    if grayscale:
        page = page.convert("L").convert("RGB")

    if fmt.lower() == "jpg":
        page.save(out_img, format="JPEG", quality=random.randint(60, 90))
    else:
        page.save(out_img, format="PNG")

    yolo_lines = []
    gt_records = []
    for p in panels:
        sb = p["struct_box"]
        lb = p["label_box"]
        struct_box = clamp_box(sb, cfg.page_w, cfg.page_h)
        yolo_lines.append(yolo_label(struct_box, cfg.page_w, cfg.page_h, class_id=0))
        if lb is not None:
            label_box = clamp_box(lb, cfg.page_w, cfg.page_h)
            yolo_lines.append(yolo_label(label_box, cfg.page_w, cfg.page_h, class_id=1))
        else:
            label_box = None
        gt_records.append({
            "struct_bbox": list(struct_box),
            "label_bbox":  list(label_box) if label_box is not None else None,
            "label_text":  p["label_text"],
            "smiles":      p["smiles"],
        })

    out_lbl.write_text("\n".join(yolo_lines))
    out_gt.write_text(json.dumps(gt_records, indent=2))


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------
# Module-level state inherited by forked workers (Linux COW-safe).
# Must stay in this module alongside _generate_one_page so pickle resolves
# the qualified name correctly across fork.
_mp_smiles: List[str] = []
_mp_fonts: List[Path] = []
_mp_distractors: List = []


def _generate_one_page(args: tuple) -> None:
    """Worker function: generate and save a single page."""
    (i, split, img_dir, lbl_dir, gt_dir, fmt,
     dpi_choices, grayscale, page_seed) = args

    random.seed(page_seed)
    np.random.seed(page_seed % (2**32))

    dpi = random.choice(dpi_choices)
    roll = random.random()

    if roll < 0.13:        # 13 % slide layouts
        cfg = make_page_config_slide(min(dpi, 200))
        page, panels = make_page(_mp_smiles, cfg, _mp_fonts, _mp_distractors)
    else:
        cfg = make_page_config(dpi)
        if roll < 0.21:    #  8 % hard negatives
            page, panels = make_negative_page(cfg, _mp_fonts, _mp_distractors)
        elif roll < 0.35:  # 14 % Excel-style tables
            page, panels = make_excel_page(_mp_smiles, cfg, _mp_fonts, _mp_distractors)
        elif roll < 0.47:  # 12 % clean compound grids
            page, panels = make_grid_page(_mp_smiles, cfg, _mp_fonts, _mp_distractors)
        elif roll < 0.55:  #  8 % single-compound data cards
            page, panels = make_data_card_page(_mp_smiles, cfg, _mp_fonts, _mp_distractors)
        elif roll < 0.63:  #  8 % SAR R-group tables
            page, panels = make_sar_page(_mp_smiles, cfg, _mp_fonts, _mp_distractors)
        elif roll < 0.70:  #  7 % matched molecular pair sheets
            page, panels = make_mmp_page(_mp_smiles, cfg, _mp_fonts, _mp_distractors)
        else:              # 30 % regular free-form pages
            page, panels = make_page(_mp_smiles, cfg, _mp_fonts, _mp_distractors)

    img_path = Path(img_dir) / f"{split}_{i:06d}.{fmt}"
    lbl_path = Path(lbl_dir) / f"{split}_{i:06d}.txt"
    gt_path  = Path(gt_dir)  / f"{split}_{i:06d}.json"
    save_sample(page, panels, img_path, lbl_path, gt_path, fmt, cfg, grayscale)


def generate_dataset(
    smiles_csv: Path,
    out_dir: Path,
    num_train: int,
    num_val: int,
    seed: int,
    fmt: str,
    fonts_dir: Optional[Path],
    distractors_dir: Optional[Path] = None,
    dpi_choices: Optional[List[int]] = None,
    grayscale: bool = False,
    workers: int = 0,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if dpi_choices is None:
        dpi_choices = [300]

    smiles_pool = load_smiles(smiles_csv)
    font_paths = find_fonts(fonts_dir)

    distractor_pool = load_distractor_images(distractors_dir)
    if distractor_pool:
        print(f"Loaded {len(distractor_pool)} distractor images from {distractors_dir}")
    else:
        print("No distractor images found — using only synthetic generators.")

    effective_workers = workers if workers > 0 else mp.cpu_count()
    print(f"DPI choices: {dpi_choices}  |  Grayscale: {grayscale}  |  Workers: {effective_workers}")

    train_img_dir = out_dir / "train" / "images"
    train_lbl_dir = out_dir / "train" / "labels"
    train_gt_dir  = out_dir / "train" / "ground_truth"
    val_img_dir   = out_dir / "val" / "images"
    val_lbl_dir   = out_dir / "val" / "labels"
    val_gt_dir    = out_dir / "val" / "ground_truth"

    for d in (train_img_dir, train_lbl_dir, train_gt_dir,
              val_img_dir, val_lbl_dir, val_gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    if effective_workers <= 1:
        def run_split(count: int, img_dir: Path, lbl_dir: Path, gt_dir: Path, split: str) -> None:
            for i in tqdm(range(count), desc=f"Generating {split}"):
                dpi = random.choice(dpi_choices)
                roll = random.random()
                if roll < 0.13:
                    cfg = make_page_config_slide(min(dpi, 200))
                    page, panels = make_page(smiles_pool, cfg, font_paths, distractor_pool)
                else:
                    cfg = make_page_config(dpi)
                    if roll < 0.21:
                        page, panels = make_negative_page(cfg, font_paths, distractor_pool)
                    elif roll < 0.35:
                        page, panels = make_excel_page(smiles_pool, cfg, font_paths, distractor_pool)
                    elif roll < 0.47:
                        page, panels = make_grid_page(smiles_pool, cfg, font_paths, distractor_pool)
                    elif roll < 0.55:
                        page, panels = make_data_card_page(smiles_pool, cfg, font_paths, distractor_pool)
                    elif roll < 0.63:
                        page, panels = make_sar_page(smiles_pool, cfg, font_paths, distractor_pool)
                    elif roll < 0.70:
                        page, panels = make_mmp_page(smiles_pool, cfg, font_paths, distractor_pool)
                    else:
                        page, panels = make_page(smiles_pool, cfg, font_paths, distractor_pool)
                img_path = img_dir / f"{split}_{i:06d}.{fmt}"
                lbl_path = lbl_dir / f"{split}_{i:06d}.txt"
                gt_path  = gt_dir  / f"{split}_{i:06d}.json"
                save_sample(page, panels, img_path, lbl_path, gt_path, fmt, cfg, grayscale)

        run_split(num_train, train_img_dir, train_lbl_dir, train_gt_dir, "train")
        run_split(num_val, val_img_dir, val_lbl_dir, val_gt_dir, "val")
    else:
        global _mp_smiles, _mp_fonts, _mp_distractors
        _mp_smiles = smiles_pool
        _mp_fonts = font_paths
        _mp_distractors = distractor_pool

        def run_split_parallel(
            count: int, img_dir: Path, lbl_dir: Path, gt_dir: Path,
            split: str, base_seed: int,
        ) -> None:
            tasks = [
                (i, split, str(img_dir), str(lbl_dir), str(gt_dir),
                 fmt, dpi_choices, grayscale, base_seed + i)
                for i in range(count)
            ]
            with mp.Pool(processes=effective_workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(_generate_one_page, tasks, chunksize=8),
                    total=count,
                    desc=f"Generating {split}",
                ):
                    pass

        run_split_parallel(num_train, train_img_dir, train_lbl_dir, train_gt_dir,
                           "train", seed)
        run_split_parallel(num_val, val_img_dir, val_lbl_dir, val_gt_dir,
                           "val", seed + num_train)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for structure detection")
    parser.add_argument("--smiles", type=Path, default=Path("data/smiles/chembl_smiles.csv"),
                        help="Path to SMILES CSV")
    parser.add_argument("--out", type=Path, default=Path("data/generated"),
                        help="Output dataset directory")
    parser.add_argument("--num-train", type=int, default=2000,
                        help="Number of training pages")
    parser.add_argument("--num-val", type=int, default=200,
                        help="Number of validation pages")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg",
                        help="Image format")
    parser.add_argument("--fonts-dir", type=Path, default=Path("data/fonts"),
                        help="Directory of .ttf/.otf fonts (default: data/fonts)")
    parser.add_argument("--distractors-dir", type=Path, default=None,
                        help="Directory of distractor images")
    parser.add_argument("--dpi", default="96,144,200,300",
                        help="Comma-separated DPI values to randomly sample per page")
    parser.add_argument("--grayscale", action="store_true", default=True,
                        help="Convert pages to grayscale before saving (default: True)")
    parser.add_argument("--no-grayscale", dest="grayscale", action="store_false")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel worker processes (0 = all CPUs, 1 = no multiprocessing)")
    args = parser.parse_args()

    if not args.smiles.exists():
        print(f"SMILES CSV not found: {args.smiles}")
        return 1

    dpi_choices = [int(d.strip()) for d in args.dpi.split(",")]

    generate_dataset(
        smiles_csv=args.smiles,
        out_dir=args.out,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed,
        fmt=args.format,
        fonts_dir=args.fonts_dir,
        distractors_dir=args.distractors_dir,
        dpi_choices=dpi_choices,
        grayscale=args.grayscale,
        workers=args.workers,
    )

    print(f"\nDone. Dataset saved under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
