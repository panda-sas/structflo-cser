"""Flask application, shared state, and all route handlers."""

import shutil
from pathlib import Path

from flask import Flask, jsonify, request, send_file, render_template

from . import pdf as pdf_mod
from . import storage

# ── App & shared state ────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates")

OUTPUT_DIR: Path = Path("data/real")   # overwritten by set_config() at startup
DPI: int = 300
PAGES: list[dict] = []                 # [{id, path, w, h}, ...]


def set_config(output_dir: Path, dpi: int) -> None:
    global OUTPUT_DIR, DPI
    OUTPUT_DIR = output_dir
    DPI = dpi


def _find(page_id: str) -> dict | None:
    return next((p for p in PAGES if p["id"] == page_id), None)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("pdf")
    if not f:
        return jsonify({"error": "no file"}), 400

    tmp = OUTPUT_DIR / "_upload.pdf"
    f.save(str(tmp))
    new_pages = pdf_mod.render_pdf(tmp, OUTPUT_DIR, DPI)
    tmp.unlink(missing_ok=True)

    existing = {p["id"] for p in PAGES}
    PAGES.extend(p for p in new_pages if p["id"] not in existing)

    return jsonify({"pages": [p["id"] for p in new_pages]})


@app.route("/pages")
def list_pages():
    result = []
    for p in PAGES:
        pairs = storage.load(p["id"], OUTPUT_DIR)
        result.append({
            "id":        p["id"],
            "w":         p["w"],
            "h":         p["h"],
            "annotated": pairs is not None,     # GT JSON exists (even if [])
            "n_pairs":   len(pairs) if pairs is not None else 0,
        })
    return jsonify(result)


@app.route("/image/<page_id>")
def serve_image(page_id):
    page = _find(page_id)
    if not page:
        return "not found", 404
    return send_file(page["path"], mimetype="image/png")


@app.route("/annotations/<page_id>", methods=["GET"])
def get_annotations(page_id):
    page = _find(page_id)
    if not page:
        return jsonify({"pairs": [], "annotated": False, "w": 0, "h": 0})
    pairs = storage.load(page_id, OUTPUT_DIR)
    return jsonify({
        "pairs":     pairs or [],
        "annotated": pairs is not None,
        "w":         page["w"],
        "h":         page["h"],
    })


@app.route("/annotations/<page_id>", methods=["POST"])
def post_annotations(page_id):
    page = _find(page_id)
    if not page:
        return jsonify({"error": "unknown page"}), 404
    data = request.get_json()
    storage.save(page_id, data["pairs"], page["w"], page["h"], OUTPUT_DIR)
    return jsonify({"saved": len(data["pairs"]), "annotated": True})


@app.route("/export", methods=["POST"])
def export():
    """Copy annotated pages from tmp/ → images/, delete tmp/, reset session.

    A page is exported if its ground-truth JSON exists (including empty pages).
    Pages never visited (no JSON) are discarded with the tmp images.
    """
    img_dir = OUTPUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    exported, skipped = 0, 0
    for page in PAGES:
        pairs = storage.load(page["id"], OUTPUT_DIR)
        if pairs is None:           # never annotated — discard
            skipped += 1
            continue
        src = Path(page["path"])    # currently in tmp/
        if src.exists():
            shutil.copy2(src, img_dir / src.name)
        exported += 1

    # Clean up tmp/
    tmp_dir = OUTPUT_DIR / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    PAGES.clear()                   # force fresh upload for next PDF

    return jsonify({
        "exported": exported,
        "skipped":  skipped,
        "images_dir": str(img_dir),
    })


@app.route("/stats")
def stats():
    pages_data = list_pages().get_json()
    annotated  = sum(1 for p in pages_data if p["annotated"])
    return jsonify({
        "pages":        len(PAGES),
        "annotated":    annotated,
        "pending":      len(PAGES) - annotated,
        "total_pairs":  sum(p["n_pairs"] for p in pages_data),
        "output_dir":   str(OUTPUT_DIR),
    })
