"""struct_labels.pipeline — extraction pipeline with adapter pattern.

Quick start
-----------
>>> from struct_labels.pipeline import ChemPipeline
>>> pipeline = ChemPipeline()
>>> pairs = pipeline.process("page.png")          # full pipeline
>>> df = ChemPipeline.to_dataframe(pairs)

Step-by-step
------------
>>> detections = pipeline.detect("page.png")
>>> pairs      = pipeline.match(detections)
>>> pairs      = pipeline.enrich(pairs, "page.png")

Custom adapters
---------------
>>> from struct_labels.pipeline import ChemPipeline, HungarianMatcher
>>> pipeline = ChemPipeline(matcher=HungarianMatcher(max_distance=500))
"""

from .matcher import BaseMatcher, HungarianMatcher
from .models import BBox, CompoundPair, Detection
from .ocr import BaseOCR, EasyOCRExtractor, NullOCR
from .pipeline import ChemPipeline
from .smiles_extractor import BaseSmilesExtractor, DecimerExtractor, NullSmilesExtractor

__all__ = [
    # Pipeline
    "ChemPipeline",
    # Data models
    "BBox",
    "Detection",
    "CompoundPair",
    # Matching adapters
    "BaseMatcher",
    "HungarianMatcher",
    # SMILES adapters
    "BaseSmilesExtractor",
    "DecimerExtractor",
    "NullSmilesExtractor",
    # OCR adapters
    "BaseOCR",
    "EasyOCRExtractor",
    "NullOCR",
]
