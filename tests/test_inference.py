"""Tests for structflo.cser.inference — NMS, tiling, pairing (no GPU needed)."""

import numpy as np

from structflo.cser.inference.nms import nms
from structflo.cser.inference.pairing import centroid, pair_detections
from structflo.cser.inference.tiling import generate_tiles


# ── NMS ──────────────────────────────────────────────────────────────────────


class TestNMS:
    def test_empty(self):
        result = nms(np.empty((0, 4)), np.array([]))
        assert len(result) == 0

    def test_single_box(self):
        boxes = np.array([[10, 10, 50, 50]])
        scores = np.array([0.9])
        keep = nms(boxes, scores)
        assert list(keep) == [0]

    def test_no_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
        scores = np.array([0.9, 0.8])
        keep = nms(boxes, scores)
        assert len(keep) == 2

    def test_high_overlap_suppresses(self):
        boxes = np.array(
            [
                [0, 0, 100, 100],
                [1, 1, 101, 101],  # almost identical
            ]
        )
        scores = np.array([0.9, 0.5])
        keep = nms(boxes, scores, iou_thresh=0.5)
        assert len(keep) == 1
        assert keep[0] == 0  # higher score wins

    def test_low_threshold_keeps_more(self):
        boxes = np.array(
            [
                [0, 0, 100, 100],
                [50, 50, 150, 150],  # partial overlap
            ]
        )
        scores = np.array([0.9, 0.8])
        # Very high threshold → keep both
        keep = nms(boxes, scores, iou_thresh=0.99)
        assert len(keep) == 2


# ── Tiling ───────────────────────────────────────────────────────────────────


class TestGenerateTiles:
    def test_small_image_single_tile(self):
        tiles = generate_tiles(500, 500, tile_size=1536)
        assert len(tiles) == 1
        assert tiles[0] == (0, 0, 500, 500)

    def test_covers_full_image(self):
        w, h = 3000, 4000
        tiles = generate_tiles(w, h, tile_size=1536, overlap=0.2)
        # Every pixel must be inside some tile
        for px in range(0, w, 100):
            for py in range(0, h, 100):
                assert any(
                    x1 <= px < x2 and y1 <= py < y2 for x1, y1, x2, y2 in tiles
                ), f"pixel ({px},{py}) not covered"

    def test_tile_size_respected(self):
        tiles = generate_tiles(5000, 5000, tile_size=1536)
        for x1, y1, x2, y2 in tiles:
            assert x2 - x1 <= 1536
            assert y2 - y1 <= 1536

    def test_overlap_creates_more_tiles(self):
        tiles_20 = generate_tiles(3000, 3000, tile_size=1536, overlap=0.2)
        tiles_50 = generate_tiles(3000, 3000, tile_size=1536, overlap=0.5)
        assert len(tiles_50) >= len(tiles_20)


# ── Pairing ──────────────────────────────────────────────────────────────────


class TestCentroid:
    def test_simple(self):
        assert centroid([0, 0, 100, 200]) == (50.0, 100.0)


class TestPairDetections:
    def _det(self, x, y, w, h, cls):
        return {"bbox": [x, y, x + w, y + h], "conf": 0.9, "class_id": cls}

    def test_empty(self):
        assert pair_detections([]) == []

    def test_no_labels(self):
        dets = [self._det(0, 0, 100, 100, 0)]
        assert pair_detections(dets) == []

    def test_no_structures(self):
        dets = [self._det(0, 0, 50, 20, 1)]
        assert pair_detections(dets) == []

    def test_one_to_one(self):
        dets = [
            self._det(0, 0, 100, 100, 0),  # structure
            self._det(110, 0, 50, 20, 1),  # label
        ]
        pairs = pair_detections(dets)
        assert len(pairs) == 1
        assert pairs[0]["structure"]["class_id"] == 0
        assert pairs[0]["label"]["class_id"] == 1

    def test_max_distance_filter(self):
        dets = [
            self._det(0, 0, 10, 10, 0),
            self._det(9000, 9000, 10, 10, 1),  # very far away
        ]
        pairs = pair_detections(dets, max_distance=100)
        assert len(pairs) == 0

    def test_closest_matching(self):
        dets = [
            self._det(0, 0, 10, 10, 0),  # struct A at ~(5,5)
            self._det(20, 0, 10, 10, 1),  # label near A at ~(25,5)
            self._det(500, 500, 10, 10, 0),  # struct B at ~(505,505)
            self._det(510, 500, 10, 10, 1),  # label near B at ~(515,505)
        ]
        pairs = pair_detections(dets)
        assert len(pairs) == 2
        # Each pair should be close
        for p in pairs:
            assert p["distance"] < 100
