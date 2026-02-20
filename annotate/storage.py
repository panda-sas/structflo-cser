"""Annotation persistence — load/save in both GT JSON and YOLO formats.

Ground-truth JSON schema:
    [
      {
        "class_id":   0,                 # 0 = chemical_structure, 1 = compound_label
        "bbox":       [x1, y1, x2, y2], # pixel coords of what the user drew
        "label_text": "",
        "smiles":     ""
      },
      ...
    ]

YOLO .txt (written only when boxes are non-empty):
    <class_id>  cx  cy  w  h   (all values normalised 0-1)

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
    """Return boxes as [{x1,y1,x2,y2,class_id}] for the canvas, or None if not annotated."""
    p = gt_path(page_id, output_dir)
    if not p.exists():
        return None                          # not yet visited
    records = json.loads(p.read_text())
    boxes = []
    for r in records:
        # Support both new schema (bbox + class_id) and legacy (union_bbox)
        if "bbox" in r:
            x1, y1, x2, y2 = r["bbox"]
        else:
            x1, y1, x2, y2 = r["union_bbox"]
        boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                       "class_id": r.get("class_id", 0)})
    return boxes


def save(page_id: str, boxes: list[dict], img_w: int, img_h: int,
         output_dir: Path) -> None:
    """Persist annotations.

    GT JSON is *always* written (even for empty pages) so the page is
    tracked as 'done'.  YOLO .txt is only written when boxes are present.
    """
    gt_dir = output_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {"class_id":   b.get("class_id", 0),
         "bbox":       [b["x1"], b["y1"], b["x2"], b["y2"]],
         "label_text": "",
         "smiles":     ""}
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
            cls = b.get("class_id", 0)
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
