"""Tests for structflo.cser.viz.detections — matplotlib visualisation helpers."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from structflo.cser.pipeline.models import BBox, CompoundPair, Detection
from structflo.cser.viz.detections import (
    _to_pil,
    plot_crops,
    plot_detections,
    plot_pairs,
    plot_results,
)

# Use non-interactive backend for CI
matplotlib.use("Agg")


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_image() -> Image.Image:
    """100×80 white test image."""
    return Image.new("RGB", (100, 80), "white")


@pytest.fixture()
def sample_detections() -> list[Detection]:
    return [
        Detection(bbox=BBox(10, 10, 40, 50), conf=0.95, class_id=0),
        Detection(bbox=BBox(60, 10, 90, 30), conf=0.88, class_id=1),
    ]


@pytest.fixture()
def sample_pairs(sample_detections: list[Detection]) -> list[CompoundPair]:
    return [
        CompoundPair(
            structure=sample_detections[0],
            label=sample_detections[1],
            match_distance=42.0,
            smiles="CCO",
            label_text="Ethanol",
        ),
    ]


@pytest.fixture()
def unenriched_pairs(sample_detections: list[Detection]) -> list[CompoundPair]:
    return [
        CompoundPair(
            structure=sample_detections[0],
            label=sample_detections[1],
            match_distance=42.0,
        ),
    ]


# ── _to_pil ─────────────────────────────────────────────────────────────────


class TestToPil:
    def test_from_pil(self, sample_image: Image.Image) -> None:
        result = _to_pil(sample_image)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_from_ndarray(self) -> None:
        arr = np.zeros((20, 30, 3), dtype=np.uint8)
        result = _to_pil(arr)
        assert isinstance(result, Image.Image)
        assert result.size == (30, 20)

    def test_from_path(self, tmp_path) -> None:
        img = Image.new("RGB", (10, 10), "red")
        path = tmp_path / "test.png"
        img.save(path)
        result = _to_pil(path)
        assert isinstance(result, Image.Image)

    def test_from_string_path(self, tmp_path) -> None:
        img = Image.new("RGB", (10, 10), "blue")
        path = tmp_path / "test.png"
        img.save(path)
        result = _to_pil(str(path))
        assert isinstance(result, Image.Image)

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported image type"):
            _to_pil(42)


# ── plot_detections ─────────────────────────────────────────────────────────


class TestPlotDetections:
    def test_returns_figure(
        self, sample_image: Image.Image, sample_detections: list[Detection]
    ) -> None:
        fig = plot_detections(sample_image, sample_detections)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_custom_title(
        self, sample_image: Image.Image, sample_detections: list[Detection]
    ) -> None:
        fig = plot_detections(sample_image, sample_detections, title="My Title")
        ax = fig.axes[0]
        assert ax.get_title() == "My Title"
        plt.close(fig)

    def test_auto_title(
        self, sample_image: Image.Image, sample_detections: list[Detection]
    ) -> None:
        fig = plot_detections(sample_image, sample_detections)
        ax = fig.axes[0]
        assert "1 structures" in ax.get_title()
        assert "1 labels" in ax.get_title()
        plt.close(fig)

    def test_empty_detections(self, sample_image: Image.Image) -> None:
        fig = plot_detections(sample_image, [])
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_with_existing_axes(
        self, sample_image: Image.Image, sample_detections: list[Detection]
    ) -> None:
        fig_ext, ax_ext = plt.subplots()
        fig = plot_detections(sample_image, sample_detections, ax=ax_ext)
        assert fig is fig_ext
        plt.close(fig)

    def test_accepts_ndarray(self, sample_detections: list[Detection]) -> None:
        arr = np.zeros((80, 100, 3), dtype=np.uint8)
        fig = plot_detections(arr, sample_detections)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ── plot_pairs ──────────────────────────────────────────────────────────────


class TestPlotPairs:
    def test_returns_figure(
        self, sample_image: Image.Image, sample_pairs: list[CompoundPair]
    ) -> None:
        fig = plot_pairs(sample_image, sample_pairs)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_custom_title(
        self, sample_image: Image.Image, sample_pairs: list[CompoundPair]
    ) -> None:
        fig = plot_pairs(sample_image, sample_pairs, title="Custom")
        assert fig.axes[0].get_title() == "Custom"
        plt.close(fig)

    def test_empty_pairs(self, sample_image: Image.Image) -> None:
        fig = plot_pairs(sample_image, [])
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ── plot_crops ──────────────────────────────────────────────────────────────


class TestPlotCrops:
    def test_returns_figure(
        self, sample_image: Image.Image, sample_pairs: list[CompoundPair]
    ) -> None:
        fig = plot_crops(sample_image, sample_pairs)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 2  # 1 pair × 2 columns
        plt.close(fig)

    def test_empty_pairs(self, sample_image: Image.Image) -> None:
        fig = plot_crops(sample_image, [])
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_multiple_pairs(self, sample_image: Image.Image) -> None:
        pairs = [
            CompoundPair(
                structure=Detection(BBox(5, 5, 20, 20), 0.9, 0),
                label=Detection(BBox(50, 5, 80, 20), 0.85, 1),
                match_distance=30.0,
            ),
            CompoundPair(
                structure=Detection(BBox(5, 40, 20, 60), 0.7, 0),
                label=Detection(BBox(50, 40, 80, 60), 0.6, 1),
                match_distance=45.0,
            ),
        ]
        fig = plot_crops(sample_image, pairs)
        assert len(fig.axes) == 4  # 2 pairs × 2 columns
        plt.close(fig)


# ── plot_results ────────────────────────────────────────────────────────────


class TestPlotResults:
    def test_returns_figure(
        self, sample_image: Image.Image, sample_pairs: list[CompoundPair]
    ) -> None:
        fig = plot_results(sample_image, sample_pairs)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_unenriched_pairs(
        self, sample_image: Image.Image, unenriched_pairs: list[CompoundPair]
    ) -> None:
        """Should handle None smiles/label_text gracefully."""
        fig = plot_results(sample_image, unenriched_pairs)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_long_smiles_truncated(self, sample_image: Image.Image) -> None:
        pair = CompoundPair(
            structure=Detection(BBox(10, 10, 40, 50), 0.9, 0),
            label=Detection(BBox(60, 10, 90, 30), 0.8, 1),
            match_distance=30.0,
            smiles="C" * 100,
            label_text="test",
        )
        fig = plot_results(sample_image, [pair], max_smiles_len=20)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_custom_title(
        self, sample_image: Image.Image, sample_pairs: list[CompoundPair]
    ) -> None:
        fig = plot_results(sample_image, sample_pairs, title="Results")
        assert fig.axes[0].get_title() == "Results"
        plt.close(fig)


# ── top-level import ────────────────────────────────────────────────────────


class TestVizInit:
    """Ensure the convenience ``from structflo.cser.viz import ...`` works."""

    def test_top_level_imports(self) -> None:
        from structflo.cser.viz import (  # noqa: F401
            plot_crops,
            plot_detections,
            plot_pairs,
            plot_results,
        )
