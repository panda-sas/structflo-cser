"""Adapter interface and implementations for OCR text extraction from label crops."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class BaseOCR(ABC):
    """Abstract interface for extracting text from a compound-label image crop.

    Implement this to swap in a different OCR engine (e.g., PaddleOCR, Tesseract)
    without touching the pipeline.
    """

    @abstractmethod
    def extract(self, image: Image.Image) -> str | None:
        """Return extracted text as a single string, or None if nothing found."""
        ...


class EasyOCRExtractor(BaseOCR):
    """Text extraction via EasyOCR (lazy-loaded, shares the PyTorch runtime with YOLO).

    EasyOCR is a good fit for compound labels: simple short alphanumeric text,
    installs with a single `pip install easyocr`, no separate framework required.
    """

    def __init__(self, languages: list[str] | None = None, gpu: bool = True) -> None:
        """
        Args:
            languages: EasyOCR language codes. Defaults to ['en'].
            gpu:       Use GPU if available (recommended; shares CUDA context with YOLO).
        """
        self._reader = None
        self.languages = languages or ["en"]
        self.gpu = gpu

    def _load(self) -> None:
        if self._reader is None:
            import easyocr  # type: ignore[import]

            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)

    def extract(self, image: Image.Image) -> str | None:
        self._load()
        try:
            result = self._reader.readtext(np.array(image), detail=0)  # type: ignore[union-attr]
            text = " ".join(result).strip()
            return text if text else None
        except Exception:
            return None


class NullOCR(BaseOCR):
    """No-op OCR — always returns None. Useful for disabling text extraction."""

    def extract(self, image: Image.Image) -> str | None:  # noqa: ARG002
        return None
