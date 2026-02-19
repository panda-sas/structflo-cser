"""PDF → PNG page rendering."""

import uuid
from pathlib import Path

import fitz  # pymupdf


def render_pdf(pdf_path: Path, output_dir: Path, dpi: int) -> list[dict]:
    """Render every page of *pdf_path* to a PNG at *dpi*, return page metadata.

    A 6-char unique suffix is appended to the stem so that re-uploading the
    same PDF never collides with a previous session's ground-truth files.

    Returns:
        list of {"id": str, "path": str (absolute), "w": int, "h": int}
    """
    img_dir = output_dir / "tmp"    # staging — moved to images/ on export
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)   # PDF unit = 1/72 inch
    uid  = uuid.uuid4().hex[:6]             # unique per upload
    stem = f"{pdf_path.stem}_{uid}"
    pages = []

    for i, page in enumerate(doc):
        pid = f"{stem}_p{i:03d}"
        out_path = img_dir / f"{pid}.png"
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        if not out_path.exists():
            pix.save(str(out_path))
        pages.append({
            "id":   pid,
            "path": str(out_path),
            "w":    pix.width,
            "h":    pix.height,
        })

    doc.close()
    return pages
