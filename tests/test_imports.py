"""Smoke tests: verify all public subpackages are importable."""


def test_import_root():
    from structflo.cser import __version__

    assert __version__


def test_import_config():
    from structflo.cser.config import PageConfig

    assert PageConfig


def test_import_geometry():
    from structflo.cser._geometry import clamp_box

    assert clamp_box


def test_import_inference_nms():
    from structflo.cser.inference.nms import nms

    assert nms


def test_import_inference_tiling():
    from structflo.cser.inference.tiling import generate_tiles

    assert generate_tiles


def test_import_inference_pairing():
    from structflo.cser.inference.pairing import pair_detections

    assert pair_detections


def test_import_pipeline_models():
    from structflo.cser.pipeline.models import BBox, CompoundPair, Detection

    assert BBox and Detection and CompoundPair


def test_import_pipeline_matcher():
    from structflo.cser.pipeline.matcher import HungarianMatcher

    assert HungarianMatcher


def test_import_pipeline_full():
    from structflo.cser.pipeline import ChemPipeline

    assert ChemPipeline
