"""Chemical structure rendering using RDKit."""

import random
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from structflo.cser._geometry import boxes_intersect
from structflo.cser.config import PageConfig


def render_structure(smiles: str, size: int, cfg: PageConfig) -> Optional[Image.Image]:
    """Render a 2-D chemical structure from a SMILES string.

    Uses RDKit's Cairo drawer to produce a transparent-background RGBA image,
    then tight-crops around the drawn atoms/bonds so there's no excess whitespace.

    Returns None if the SMILES is invalid or produces no visible drawing.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        return None

    drawer = rdMolDraw2D.MolDraw2DCairo(size, size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = random.uniform(*cfg.bond_width_range)
    opts.minFontSize = random.randint(*cfg.atom_font_range)
    opts.maxFontSize = opts.minFontSize + 8
    opts.additionalAtomLabelPadding = random.uniform(0.05, 0.2)
    opts.rotate = random.uniform(0, 360)
    if random.random() < 0.3:
        opts.useBWAtomPalette()

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    img = Image.open(BytesIO(drawer.GetDrawingText())).convert("RGBA")
    arr = np.array(img)

    # Tight-crop: keep only non-transparent, non-white pixels
    mask = (arr[:, :, 3] > 0) & (
        (arr[:, :, 0] < 250) | (arr[:, :, 1] < 250) | (arr[:, :, 2] < 250)
    )
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
    return cropped


def place_structure(
    page: Image.Image,
    struct_img: Image.Image,
    cfg: PageConfig,
    existing_boxes: List[Tuple[int, int, int, int]],
    max_tries: int = 80,
    x_range: Optional[Tuple[int, int]] = None,
    y_range: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """Try to paste *struct_img* onto *page* without overlapping existing boxes.

    Randomly samples positions within the allowed region (x_range/y_range or
    page margins) up to *max_tries* times. Returns the (x0,y0,x1,y1) box on
    success, or None if no free position was found.
    """
    w, h = page.size
    sw, sh = struct_img.size

    x_lo = x_range[0] if x_range else cfg.margin
    x_hi = (x_range[1] - sw) if x_range else (w - cfg.margin - sw)
    y_lo = y_range[0] if y_range else cfg.margin
    y_hi = (y_range[1] - sh) if y_range else (h - cfg.margin - sh)

    if x_lo >= x_hi or y_lo >= y_hi:
        return None

    for _ in range(max_tries):
        x = random.randint(x_lo, x_hi)
        y = random.randint(y_lo, y_hi)
        box = (x, y, x + sw, y + sh)
        padded = (x - 6, y - 6, x + sw + 6, y + sh + 6)

        if any(boxes_intersect(padded, b) for b in existing_boxes):
            continue

        page.paste(struct_img, (x, y), struct_img)
        return box
    return None
