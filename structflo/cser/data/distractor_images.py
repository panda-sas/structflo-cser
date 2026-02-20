"""Distractor image loading and Picsum download utilities."""

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import requests
from PIL import Image
from tqdm import tqdm

PICSUM_LIST_URL = "https://picsum.photos/v2/list"


def load_distractor_images(distractors_dir: Optional[Path]) -> List[Image.Image]:
    """Pre-load distractor images from disk (kept at original size for variety)."""
    if distractors_dir is None or not distractors_dir.exists():
        return []
    imgs: List[Image.Image] = []
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(distractors_dir.glob(ext))
    for p in sorted(paths):
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(img)
        except Exception:
            continue
    return imgs


def _pick_distractor_image(distractor_pool: List[Image.Image]) -> Image.Image:
    """Pick a random real distractor image and resize it to a random distractor size."""
    img = random.choice(distractor_pool).copy()
    target_w = random.randint(150, 500)
    target_h = random.randint(120, 400)
    img = img.resize((target_w, target_h), Image.LANCZOS)
    if random.random() < 0.25:
        img = img.convert("L").convert("RGB")
    return img


def fetch_picsum_list(n_pages: int = 30) -> list:
    """Fetch the Picsum photo list (paginated, up to 100 per page)."""
    all_items: list = []
    for page in range(1, n_pages + 1):
        resp = requests.get(
            PICSUM_LIST_URL,
            params={"page": page, "limit": 100},
            timeout=30,
        )
        if resp.status_code != 200:
            break
        items = resp.json()
        if not items:
            break
        all_items.extend(items)
    return all_items


def download_picsum(
    photo_id: str, width: int, height: int, out_path: Path, timeout: int = 20
) -> bool:
    """Download a specific Picsum photo at a given size."""
    url = f"https://picsum.photos/id/{photo_id}/{width}/{height}"
    try:
        resp = requests.get(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        ct = resp.headers.get("Content-Type", "")
        if "image" not in ct:
            return False
        if len(resp.content) < 1000:
            return False
        out_path.write_bytes(resp.content)
        return True
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download distractor images for synthetic page generation"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/distractors"),
        help="Output directory for downloaded images",
    )
    parser.add_argument(
        "--count", type=int, default=300, help="Number of images to download"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible selection"
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of parallel download threads"
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
    if len(existing) >= args.count:
        print(f"Already have {len(existing)} images in {out_dir}, skipping download.")
        return 0

    needed = args.count - len(existing)
    idx_start = len(existing)

    print("Fetching photo list from Lorem Picsum (Unsplash)...")
    photos = fetch_picsum_list(n_pages=30)
    if not photos:
        print("ERROR: Could not fetch photo list from Picsum.")
        return 1
    print(f"  Found {len(photos)} photos available.")

    rng = random.Random(args.seed)
    sizes = [
        (300, 200),
        (400, 300),
        (350, 250),
        (250, 180),
        (500, 350),
        (200, 300),
        (280, 280),
        (450, 200),
        (320, 240),
        (380, 260),
        (220, 330),
        (480, 320),
    ]

    tasks: list = []
    while len(tasks) < needed:
        rng.shuffle(photos)
        for p in photos:
            w, h = rng.choice(sizes)
            tasks.append((p["id"], w, h))
            if len(tasks) >= needed:
                break

    print(f"Downloading {needed} images with {args.workers} threads...")
    downloaded = 0
    failed = 0
    idx = idx_start

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for photo_id, w, h in tasks:
            out_path = out_dir / f"distractor_{idx:05d}.jpg"
            idx += 1
            futures[pool.submit(download_picsum, photo_id, w, h, out_path)] = out_path

        pbar = tqdm(total=needed, desc="Downloading")
        for future in as_completed(futures):
            out_path = futures[future]
            try:
                success = future.result()
            except Exception:
                success = False

            if success:
                downloaded += 1
                pbar.update(1)
            else:
                failed += 1
                out_path.unlink(missing_ok=True)
        pbar.close()

    for f in out_dir.glob("distractor_*.jpg"):
        if f.stat().st_size == 0:
            f.unlink()

    final_count = len(list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png")))
    print(f"\nDone. {final_count} distractor images in {out_dir}")
    print(f"  (downloaded: {downloaded}, failed: {failed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
