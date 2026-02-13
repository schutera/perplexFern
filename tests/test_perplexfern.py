"""Tests for the perplexfern package (v0.3 – multi-scale)."""

import json
import csv
import numpy as np
import pytest

from perplexfern import parser, metrics, fractal, visualize, galaxy
from perplexfern import galaxy_viz
import perplexfern


SAMPLE = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
) * 20  # enough text for multi-scale analysis

SHORT = "Hello world hello world hello again."


# ── Parser ──────────────────────────────────────────────────────────────────

class TestParser:
    def test_raw_string(self):
        assert parser.parse(SAMPLE) == SAMPLE

    def test_txt_file(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_text(SAMPLE, encoding="utf-8")
        assert parser.parse(str(p)) == SAMPLE

    def test_json_file(self, tmp_path):
        p = tmp_path / "test.json"
        data = {"title": "Hello", "body": "World"}
        p.write_text(json.dumps(data), encoding="utf-8")
        result = parser.parse(str(p))
        assert "Hello" in result and "World" in result

    def test_csv_file(self, tmp_path):
        p = tmp_path / "test.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["a", "b"], ["c", "d"]])
        assert "a" in parser.parse(str(p))


# ── Windowed metrics ────────────────────────────────────────────────────────

class TestWindowedMetrics:
    def test_returns_array(self):
        sig = metrics.compute_windowed(SAMPLE, metric="perplexity")
        assert isinstance(sig, np.ndarray) and sig.ndim == 1

    def test_all_metrics_run(self):
        for name in metrics.AVAILABLE_METRICS:
            sig = metrics.compute_windowed(SAMPLE, metric=name)
            assert len(sig) >= 1

    def test_unknown_metric(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            metrics.compute_windowed(SAMPLE, metric="nonexistent")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            metrics.compute_windowed("")

    def test_short_text(self):
        sig = metrics.compute_windowed(SHORT, metric="perplexity")
        assert len(sig) >= 1


# ── Fractal (multi-scale) ──────────────────────────────────────────────────

class TestFractal:
    def test_multi_scale_shape(self):
        fn = metrics.AVAILABLE_METRICS["perplexity"]
        img = fractal.multi_scale_image(SAMPLE, fn, order=5, n_octaves=3)
        assert img.shape == (32, 32)

    def test_multi_scale_values_01(self):
        fn = metrics.AVAILABLE_METRICS["perplexity"]
        img = fractal.multi_scale_image(SAMPLE, fn, order=5, n_octaves=3)
        assert img.min() >= 0.0 - 1e-9
        assert img.max() <= 1.0 + 1e-9

    def test_signal_to_image_compat(self):
        sig = np.random.default_rng(0).random(100)
        img = fractal.signal_to_image(sig)
        assert img.ndim == 2 and img.shape[0] == img.shape[1]

    def test_deterministic(self):
        fn = metrics.AVAILABLE_METRICS["entropy"]
        a = fractal.multi_scale_image(SAMPLE, fn, order=4, n_octaves=2)
        b = fractal.multi_scale_image(SAMPLE, fn, order=4, n_octaves=2)
        np.testing.assert_array_equal(a, b)


# ── Visualize ───────────────────────────────────────────────────────────────

class TestVisualize:
    def test_render_returns_figure(self):
        img = np.random.default_rng(0).random((16, 16))
        fig = visualize.render(img)
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_save(self, tmp_path):
        img = np.random.default_rng(0).random((16, 16))
        out = tmp_path / "test.png"
        visualize.render(img, save_path=out)
        assert out.exists() and out.stat().st_size > 0
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_render_to_bytes_png(self):
        img = np.random.default_rng(0).random((16, 16))
        data = visualize.render_to_bytes(img, fmt="png")
        assert data[:4] == b"\x89PNG"


# ── End-to-end API ──────────────────────────────────────────────────────────

class TestAPI:
    def test_analyse(self, tmp_path):
        out = tmp_path / "fern.png"
        fig = perplexfern.analyse(
            SAMPLE, save_path=out, metric="perplexity",
            order=4, n_octaves=2,
        )
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        assert out.exists()
        plt.close(fig)

    def test_analyse_metrics(self):
        m = perplexfern.analyse_metrics(SAMPLE)
        assert isinstance(m, metrics.TextMetrics)
        assert m.perplexity > 0


# ── Galaxy module ───────────────────────────────────────────────────────────

class TestGalaxy:
    def test_galaxy_points_shapes(self):
        metric_dict = {n: f for n, f in sorted(metrics.AVAILABLE_METRICS.items())}
        pos, feat = galaxy.galaxy_points(SAMPLE, metric_dict)
        assert pos.ndim == 2 and pos.shape[1] == 2
        assert feat.ndim == 2 and feat.shape[1] == len(metric_dict)
        assert pos.shape[0] == feat.shape[0]

    def test_galaxy_image_shape_and_range(self):
        metric_dict = {n: f for n, f in sorted(metrics.AVAILABLE_METRICS.items())}
        pos, feat = galaxy.galaxy_points(SAMPLE, metric_dict, window_chars=60, stride_chars=30)
        img = galaxy.galaxy_image(
            pos, feat, list(metric_dict.keys()), resolution=256,
        )
        assert img.shape == (256, 256, 4)
        assert img.min() >= 0.0 - 1e-6
        assert img.max() <= 1.0 + 1e-6

    def test_galaxy_image_with_glow(self):
        metric_dict = {n: f for n, f in sorted(metrics.AVAILABLE_METRICS.items())}
        pos, feat = galaxy.galaxy_points(SAMPLE, metric_dict, window_chars=60, stride_chars=30)
        img = galaxy.galaxy_image(
            pos, feat, list(metric_dict.keys()),
            resolution=64, glow_sigma=3.0,
        )
        assert img.shape == (64, 64, 4)

    def test_render_galaxy_figure(self):
        img = np.random.default_rng(0).random((32, 32, 4)).astype(np.float32)
        fig = galaxy_viz.render_galaxy(img)
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_galaxy_save(self, tmp_path):
        img = np.random.default_rng(0).random((32, 32, 4)).astype(np.float32)
        out = tmp_path / "galaxy.png"
        galaxy_viz.render_galaxy(img, save_path=out)
        assert out.exists() and out.stat().st_size > 0
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_analyse_galaxy_e2e(self, tmp_path):
        out = tmp_path / "galaxy_e2e.png"
        fig = perplexfern.analyse_galaxy(
            SAMPLE, save_path=out, resolution=64, glow=0,
        )
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)
        assert out.exists()
        plt.close(fig)

    def test_custom_palette(self):
        metric_dict = {n: f for n, f in sorted(metrics.AVAILABLE_METRICS.items())}
        pos, feat = galaxy.galaxy_points(SAMPLE, metric_dict, window_chars=60, stride_chars=30)
        img = galaxy.galaxy_image(
            pos, feat, list(metric_dict.keys()),
            resolution=64, palette="plasma",
        )
        assert img.shape == (64, 64, 4)
