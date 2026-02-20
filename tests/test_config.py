"""Tests for structflo.cser.config — PageConfig dataclass and factories."""

import pytest

from structflo.cser.config import PageConfig, make_page_config, make_page_config_slide


class TestPageConfig:
    def test_defaults(self):
        cfg = PageConfig()
        assert cfg.page_w == 2480
        assert cfg.page_h == 3508
        assert cfg.margin == 180
        assert cfg.min_structures >= 1

    def test_custom_values(self):
        cfg = PageConfig(page_w=800, page_h=600, margin=20)
        assert cfg.page_w == 800
        assert cfg.page_h == 600


class TestMakePageConfig:
    def test_300dpi_matches_defaults(self):
        cfg = make_page_config(300)
        assert cfg.page_w == 2480
        assert cfg.page_h == 3508

    def test_lower_dpi_shrinks(self):
        cfg = make_page_config(150)
        assert cfg.page_w < 2480
        assert cfg.page_h < 3508

    def test_scaling_ratio(self):
        cfg_300 = make_page_config(300)
        cfg_150 = make_page_config(150)
        assert cfg_150.page_w == pytest.approx(cfg_300.page_w / 2, abs=1)


class TestMakePageConfigSlide:
    def test_landscape(self):
        cfg = make_page_config_slide(96)
        assert cfg.page_w > cfg.page_h  # landscape

    def test_fewer_structures(self):
        cfg = make_page_config_slide()
        assert cfg.max_structures <= 6
