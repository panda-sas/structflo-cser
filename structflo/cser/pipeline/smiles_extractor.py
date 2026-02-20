"""Adapter interface and implementations for SMILES extraction from structure crops."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class BaseSmilesExtractor(ABC):
    """Abstract interface for extracting a SMILES string from a chemical structure image.

    Implement this to swap in a different model (e.g., MolScribe, OSRA) without
    touching the pipeline.
    """

    @abstractmethod
    def extract(self, image: Image.Image) -> str | None:
        """Return a SMILES string, or None if extraction fails or produces no result."""
        ...


class DecimerExtractor(BaseSmilesExtractor):
    """SMILES extraction using the DECIMER deep-learning model (lazy-loaded).

    predict_SMILES accepts either a file path (str) or a numpy array directly,
    so we convert the PIL crop to ndarray and avoid any temp-file overhead.
    """

    def __init__(self) -> None:
        self._predict = None  # loaded on first call

    def _load(self) -> None:
        if self._predict is None:
            from DECIMER import predict_SMILES  # type: ignore[import]

            self._predict = predict_SMILES

    def extract(self, image: Image.Image) -> str | None:
        self._load()
        try:
            img_array = np.array(image.convert("RGB"))
            smiles: str = self._predict(img_array)  # type: ignore[misc]
            return smiles if smiles else None
        except Exception:
            return None


class NullSmilesExtractor(BaseSmilesExtractor):
    """No-op extractor — always returns None. Useful for disabling SMILES extraction."""

    def extract(self, image: Image.Image) -> str | None:  # noqa: ARG002
        return None
