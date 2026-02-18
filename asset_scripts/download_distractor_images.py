#!/usr/bin/env python3
"""
Download diverse distractor images for synthetic page generation.

Downloads random real-world photos from Lorem Picsum (backed by Unsplash)
to use as hard-negative distractors. These help the YOLO model learn NOT
to fire on non-chemistry images embedded in document pages.

Usage:
    python scripts/download_distractor_images.py --out data/distractors --count 300
"""

import argparse
import random
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

# Lorem Picsum – serves real photos from Unsplash. No API key needed.
PICSUM_LIST_URL = "https://picsum.photos/v2/list"


def fetch_picsum_list(n_pages: int = 30) -> list[dict]:
    """Fetch the Picsum photo list (paginated, up to 100 per page)."""
    all_items: list[dict] = []
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
        "--count",
        type=int,
        default=300,
        help="Number of images to download",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible selection",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download threads",
    )
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check how many we already have
    existing = list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png"))
    if len(existing) >= args.count:
        print(f"Already have {len(existing)} images in {out_dir}, skipping download.")
        return 0

    needed = args.count - len(existing)
    idx_start = len(existing)

    # Fetch available photo IDs from Picsum
    print("Fetching photo list from Lorem Picsum (Unsplash)...")
    photos = fetch_picsum_list(n_pages=30)
    if not photos:
        print("ERROR: Could not fetch photo list from Picsum.")
        return 1
    print(f"  Found {len(photos)} photos available.")

    rng = random.Random(args.seed)

    # Build download tasks – vary sizes to simulate different document insets
    sizes = [
        (300, 200), (400, 300), (350, 250), (250, 180),
        (500, 350), (200, 300), (280, 280), (450, 200),
        (320, 240), (380, 260), (220, 330), (480, 320),
    ]

    # Repeat / shuffle photos to fill `needed`
    tasks: list[tuple[str, int, int]] = []
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

    # Clean up any empty files
    for f in out_dir.glob("distractor_*.jpg"):
        if f.stat().st_size == 0:
            f.unlink()

    final_count = len(list(out_dir.glob("*.jpg")) + list(out_dir.glob("*.png")))
    print(f"\nDone. {final_count} distractor images in {out_dir}")
    print(f"  (downloaded: {downloaded}, failed: {failed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
