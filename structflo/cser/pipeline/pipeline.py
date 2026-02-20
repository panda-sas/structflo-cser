"""ChemPipeline: detect → match → extract SMILES + OCR text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from structflo.cser.inference.detector import DEFAULT_WEIGHTS, detect_full, detect_tiled

from structflo.cser.pipeline.matcher import BaseMatcher, HungarianMatcher
from structflo.cser.pipeline.models import BBox, CompoundPair, Detection
from structflo.cser.pipeline.ocr import BaseOCR, EasyOCRExtractor
from structflo.cser.pipeline.smiles_extractor import (
    BaseSmilesExtractor,
    DecimerExtractor,
)

# Anything the pipeline accepts as an image input
ImageLike = Union[Path, str, np.ndarray, Image.Image]


def _to_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


class ChemPipeline:
    """End-to-end pipeline from an image to enriched (SMILES, label-text) pairs.

    Designed after the HuggingFace transformers pattern: every step is exposed
    individually for fine-grained control, and a single ``process()`` call runs
    the whole thing for convenience.

    Low-level access
    ----------------
    >>> detections = pipeline.detect(image)
    >>> pairs      = pipeline.match(detections)
    >>> smiles     = pipeline.extract_smiles(image, pair)
    >>> text       = pipeline.extract_text(image, pair)
    >>> pairs      = pipeline.enrich(pairs, image)

    High-level access
    -----------------
    >>> pairs = pipeline.process("page.png")
    >>> df    = ChemPipeline.to_dataframe(pairs)
    >>> data  = ChemPipeline.to_records(pairs)

    Adapter pattern
    ---------------
    Pass custom implementations of ``BaseMatcher``, ``BaseSmilesExtractor``, or
    ``BaseOCR`` to swap out any step without modifying this class.
    """

    def __init__(
        self,
        *,
        weights: Path | str | None = None,
        matcher: BaseMatcher | None = None,
        smiles_extractor: BaseSmilesExtractor | None = None,
        ocr: BaseOCR | None = None,
        tile: bool = True,
        tile_size: int = 1536,
        conf: float = 0.3,
    ) -> None:
        """
        Args:
            weights:          Path to YOLO .pt weights file.  Defaults to the
                              trained model in ``runs/labels_detect/``.
            matcher:          Pairing strategy.  Defaults to HungarianMatcher.
            smiles_extractor: SMILES model.  Defaults to DecimerExtractor.
            ocr:              OCR engine.  Defaults to PaddleOCRExtractor.
            tile:             Use sliding-window tiling during detection.
            tile_size:        Tile side length in pixels.
            conf:             YOLO confidence threshold.
        """
        self._weights_path = Path(weights) if weights else DEFAULT_WEIGHTS
        self._matcher = matcher or HungarianMatcher()
        self._smiles = smiles_extractor or DecimerExtractor()
        self._ocr = ocr or EasyOCRExtractor()
        self.tile = tile
        self.tile_size = tile_size
        self.conf = conf
        self._model = None  # ultralytics YOLO — lazy-loaded on first detect() call

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self._weights_path))

    @staticmethod
    def _crop(image: Image.Image, bbox: BBox) -> Image.Image:
        return image.crop((int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)))

    # ------------------------------------------------------------------
    # Low-level step methods
    # ------------------------------------------------------------------

    def detect(self, image: ImageLike) -> list[Detection]:
        """Run YOLO on *image* and return a flat list of Detection objects.

        Both ``structure`` (class 0) and ``label`` (class 1) detections are
        returned together; call ``match()`` next to pair them.
        """
        self._load_model()
        img_pil = _to_pil(image)
        img_np = np.array(img_pil)
        if self.tile:
            raw = detect_tiled(
                self._model, img_np, tile_size=self.tile_size, conf=self.conf
            )
        else:
            raw = detect_full(self._model, img_np, conf=self.conf)
        return [Detection.from_dict(d) for d in raw]

    def match(self, detections: list[Detection]) -> list[CompoundPair]:
        """Pair structure detections with label detections using the configured matcher."""
        return self._matcher.match(detections)

    def extract_smiles(self, image: ImageLike, pair: CompoundPair) -> str | None:
        """Crop the structure region from *image* and extract a SMILES string."""
        img = _to_pil(image)
        crop = self._crop(img, pair.structure.bbox)
        return self._smiles.extract(crop)

    def extract_text(self, image: ImageLike, pair: CompoundPair) -> str | None:
        """Crop the label region from *image* and extract text via OCR."""
        img = _to_pil(image)
        crop = self._crop(img, pair.label.bbox)
        return self._ocr.extract(crop)

    def enrich(self, pairs: list[CompoundPair], image: ImageLike) -> list[CompoundPair]:
        """Populate ``smiles`` and ``label_text`` on every pair in-place.

        The image is decoded once and reused for all crops.  Returns the same
        list for convenience.
        """
        img = _to_pil(image)
        for pair in pairs:
            pair.smiles = self._smiles.extract(self._crop(img, pair.structure.bbox))
            pair.label_text = self._ocr.extract(self._crop(img, pair.label.bbox))
        return pairs

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    def process(self, image: ImageLike) -> list[CompoundPair]:
        """Full pipeline in one call: detect → match → enrich.

        Returns a list of CompoundPair objects with ``smiles`` and
        ``label_text`` populated.
        """
        img = _to_pil(image)
        detections = self.detect(img)
        pairs = self.match(detections)
        return self.enrich(pairs, img)

    # ------------------------------------------------------------------
    # Output helpers  (static — can also be called on the class directly)
    # ------------------------------------------------------------------

    @staticmethod
    def to_records(pairs: list[CompoundPair]) -> list[dict]:
        """Serialise pairs to a list of plain dicts (JSON-serialisable)."""
        return [p.to_dict() for p in pairs]

    @staticmethod
    def to_json(pairs: list[CompoundPair], indent: int = 2) -> str:
        """Serialise pairs to a formatted JSON string."""
        return json.dumps(ChemPipeline.to_records(pairs), indent=indent)

    @staticmethod
    def to_dataframe(pairs: list[CompoundPair]):
        """Convert pairs to a pandas DataFrame.

        Requires pandas to be installed (``pip install pandas``).
        """
        import pandas as pd  # type: ignore[import]

        return pd.DataFrame(ChemPipeline.to_records(pairs))
