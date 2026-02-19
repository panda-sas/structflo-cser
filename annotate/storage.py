"""Annotation persistence — load/save in both GT JSON and YOLO formats.

Ground-truth JSON (matches synthetic data schema):
    [
      {
        "union_bbox":  [x1, y1, x2, y2],   # what the user drew
        "struct_bbox": null,                 # not separately annotated in real data
        "label_bbox":  null,
        "label_text":  "",
        "smiles":      ""
      },
      ...
    ]

YOLO .txt (written only when boxes are non-empty):
    0  cx  cy  w  h   (all values normalised 0-1)

Annotation states:
    - GT JSON absent  → page not yet visited
    - GT JSON = []    → page explicitly marked as "no panels" (empty page)
    - GT JSON = [...]  → page annotated with N boxes
"""

import json
from pathlib import Path


def gt_path(page_id: str, output_dir: Path) -> Path:
    return output_dir / "ground_truth" / f"{page_id}.json"


def lbl_path(page_id: str, output_dir: Path) -> Path:
    return output_dir / "labels" / f"{page_id}.txt"


def load(page_id: str, output_dir: Path) -> list[dict] | None:
    """Return boxes as [{x1,y1,x2,y2}] for the canvas, or None if not annotated."""
    p = gt_path(page_id, output_dir)
    if not p.exists():
        return None                          # not yet visited
    records = json.loads(p.read_text())
    return [
        {"x1": r["union_bbox"][0], "y1": r["union_bbox"][1],
         "x2": r["union_bbox"][2], "y2": r["union_bbox"][3]}
        for r in records
    ]


def save(page_id: str, boxes: list[dict], img_w: int, img_h: int,
         output_dir: Path) -> None:
    """Persist annotations.

    GT JSON is *always* written (even for empty pages) so the page is
    tracked as 'done'.  YOLO .txt is only written when boxes are present.
    """
    gt_dir = output_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {"union_bbox":  [b["x1"], b["y1"], b["x2"], b["y2"]],
         "struct_bbox": None,
         "label_bbox":  None,
         "label_text":  "",
         "smiles":      ""}
        for b in boxes
    ]
    gt_path(page_id, output_dir).write_text(json.dumps(records, indent=2))

    lbl = lbl_path(page_id, output_dir)
    if not boxes:
        lbl.unlink(missing_ok=True)         # no YOLO file for empty pages
        return

    lbl.parent.mkdir(parents=True, exist_ok=True)
    with open(lbl, "w") as f:
        for b in boxes:
            x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
