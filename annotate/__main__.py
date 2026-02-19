#!/usr/bin/env python3
"""
Compound panel annotation tool — entry point.

Usage (from project root):
    python annotate/__main__.py
    python annotate/__main__.py --out data/real --port 8000
    python -m annotate
"""

import argparse
import sys
from pathlib import Path

# Allow both `python annotate/__main__.py` and `python -m annotate`
sys.path.insert(0, str(Path(__file__).parent.parent))

from annotate.server import app, set_config   # noqa: E402


def main() -> None:
    _default_out = (Path(__file__).parent.parent / "data" / "real").resolve()

    p = argparse.ArgumentParser(description="Compound panel annotation tool")
    p.add_argument("--out", default=str(_default_out),
                   help="Output directory (default: <project>/data/real)")
    p.add_argument("--dpi", type=int, default=300,
                   help="PDF render DPI — 300 matches synthetic training data")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_config(output_dir, args.dpi)

    print(f"Annotator:  http://{args.host}:{args.port}")
    print(f"Output dir: {output_dir}")
    print(f"DPI:        {args.dpi}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
