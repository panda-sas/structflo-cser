"""Tests for structflo.cser._geometry — bbox helpers."""

from structflo.cser._geometry import boxes_intersect, clamp_box, try_place_box


# ── clamp_box ────────────────────────────────────────────────────────────────


class TestClampBox:
    def test_within_bounds(self):
        assert clamp_box((10, 20, 100, 200), 300, 400) == (10, 20, 100, 200)

    def test_clamp_negative(self):
        assert clamp_box((-5, -10, 50, 60), 200, 200) == (0, 0, 50, 60)

    def test_clamp_overflow(self):
        assert clamp_box((10, 10, 500, 600), 200, 200) == (10, 10, 200, 200)

    def test_clamp_both_sides(self):
        x0, y0, x1, y1 = clamp_box((-50, -50, 9999, 9999), 100, 100)
        assert (x0, y0, x1, y1) == (0, 0, 100, 100)


# ── boxes_intersect ──────────────────────────────────────────────────────────


class TestBoxesIntersect:
    def test_overlap(self):
        assert boxes_intersect((0, 0, 10, 10), (5, 5, 15, 15)) is True

    def test_no_overlap_horizontal(self):
        assert boxes_intersect((0, 0, 10, 10), (20, 0, 30, 10)) is False

    def test_no_overlap_vertical(self):
        assert boxes_intersect((0, 0, 10, 10), (0, 20, 10, 30)) is False

    def test_touching_edges(self):
        """Touching edges (shared boundary) do NOT count as overlap."""
        assert boxes_intersect((0, 0, 10, 10), (10, 0, 20, 10)) is False

    def test_containment(self):
        assert boxes_intersect((0, 0, 100, 100), (10, 10, 20, 20)) is True

    def test_identical(self):
        assert boxes_intersect((5, 5, 15, 15), (5, 5, 15, 15)) is True


# ── try_place_box ────────────────────────────────────────────────────────────


class TestTryPlaceBox:
    def test_returns_box_in_empty_space(self):
        box = try_place_box(1000, 1000, 50, 50, margin=10, existing=[])
        assert box is not None
        x0, y0, x1, y1 = box
        assert x1 - x0 == 50
        assert y1 - y0 == 50

    def test_returns_none_when_full(self):
        """A tiny page completely covered → should fail placement."""
        existing = [(0, 0, 100, 100)]
        result = try_place_box(
            100, 100, 80, 80, margin=0, existing=existing, max_tries=100
        )
        assert result is None

    def test_no_overlap_with_existing(self):
        existing = [(0, 0, 500, 500)]
        box = try_place_box(
            1000, 1000, 50, 50, margin=10, existing=existing, max_tries=200
        )
        if box is not None:
            assert not boxes_intersect(box, existing[0])
