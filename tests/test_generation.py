"""Tests for structflo.cser.generation.dataset — YOLO label formatting."""

import pytest

from structflo.cser.generation.dataset import yolo_label


class TestYoloLabel:
    def test_format_center(self):
        """A box covering the full page → center at (0.5, 0.5), size (1.0, 1.0)."""
        line = yolo_label((0, 0, 200, 400), w=200, h=400)
        parts = line.split()
        assert parts[0] == "0"  # class_id
        assert float(parts[1]) == pytest.approx(0.5)
        assert float(parts[2]) == pytest.approx(0.5)
        assert float(parts[3]) == pytest.approx(1.0)
        assert float(parts[4]) == pytest.approx(1.0)

    def test_class_id(self):
        line = yolo_label((10, 10, 50, 50), w=100, h=100, class_id=1)
        assert line.startswith("1 ")

    def test_small_box(self):
        line = yolo_label((45, 45, 55, 55), w=100, h=100)
        parts = line.split()
        assert float(parts[1]) == pytest.approx(0.5)  # cx
        assert float(parts[2]) == pytest.approx(0.5)  # cy
        assert float(parts[3]) == pytest.approx(0.1)  # bw
        assert float(parts[4]) == pytest.approx(0.1)  # bh

    def test_output_has_six_fields(self):
        line = yolo_label((0, 0, 10, 20), w=100, h=100)
        assert len(line.split()) == 5  # class cx cy bw bh
