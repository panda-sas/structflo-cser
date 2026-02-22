"""Tests for structflo.cser.pipeline.models — BBox, Detection, CompoundPair."""

import pytest

from structflo.cser.pipeline.models import BBox, CompoundPair, Detection


# ── BBox ─────────────────────────────────────────────────────────────────────


class TestBBox:
    def test_from_list(self):
        b = BBox.from_list([10.0, 20.0, 110.0, 120.0])
        assert b.x1 == 10.0 and b.y1 == 20.0
        assert b.x2 == 110.0 and b.y2 == 120.0

    def test_centroid(self):
        b = BBox(0, 0, 100, 200)
        assert b.centroid == (50.0, 100.0)

    def test_width_height(self):
        b = BBox(10, 20, 60, 80)
        assert b.width == 50
        assert b.height == 60

    def test_as_list_roundtrip(self):
        original = [1.5, 2.5, 3.5, 4.5]
        b = BBox.from_list(original)
        assert b.as_list() == original

    def test_as_tuple(self):
        b = BBox(1, 2, 3, 4)
        assert b.as_tuple() == (1, 2, 3, 4)


# ── Detection ────────────────────────────────────────────────────────────────


class TestDetection:
    def test_from_dict(self):
        d = Detection.from_dict(
            {"bbox": [10, 20, 100, 200], "conf": 0.95, "class_id": 0}
        )
        assert d.class_id == 0
        assert d.conf == 0.95
        assert d.bbox.x1 == 10

    def test_class_ids(self):
        struct = Detection(BBox(0, 0, 1, 1), 0.9, class_id=0)
        label = Detection(BBox(0, 0, 1, 1), 0.8, class_id=1)
        assert struct.class_id == 0  # structure
        assert label.class_id == 1  # label


# ── CompoundPair ─────────────────────────────────────────────────────────────


class TestCompoundPair:
    @pytest.fixture()
    def pair(self):
        return CompoundPair(
            structure=Detection(BBox(0, 0, 100, 100), 0.95, 0),
            label=Detection(BBox(110, 0, 160, 30), 0.88, 1),
            match_distance=15.0,
            smiles="CCO",
            label_text="Compound-1",
        )

    def test_to_dict_keys(self, pair):
        d = pair.to_dict()
        expected_keys = {
            "structure_bbox",
            "structure_conf",
            "label_bbox",
            "label_conf",
            "match_distance",
            "match_confidence",
            "smiles",
            "label_text",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self, pair):
        d = pair.to_dict()
        assert d["smiles"] == "CCO"
        assert d["label_text"] == "Compound-1"
        assert d["match_distance"] == 15.0

    def test_optional_fields_default_none(self):
        p = CompoundPair(
            structure=Detection(BBox(0, 0, 1, 1), 0.5, 0),
            label=Detection(BBox(2, 2, 3, 3), 0.5, 1),
            match_distance=10.0,
        )
        assert p.smiles is None
        assert p.label_text is None
