"""
perplexfern – Visualise the complexity of natural text as a fractal.

Computes a metric (perplexity, entropy, etc.) over sliding windows at
multiple scales, then maps the blended signal onto a square image via
a Hilbert curve.  The fractal structure emerges from the statistical
self-similarity of the language itself.

Quick start
-----------
>>> import perplexfern
>>> fig = perplexfern.analyse("Your text here …", save_path="fractal.png")
"""

from __future__ import annotations

__version__ = "0.3.0"

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from .metrics import (
    AVAILABLE_METRICS,
    TextMetrics,
    compute_summary,
    compute_windowed,
)
from .parser import parse
from .fractal import signal_to_image, multi_scale_image
from .visualize import render, render_to_bytes
from .galaxy import galaxy_points, galaxy_image
from .galaxy_viz import render_galaxy, render_galaxy_to_bytes


# Sensible default metric ranges for intensity anchoring.
# (low_value, high_value) – values outside are clamped.
_METRIC_RANGES: dict[str, tuple[float, float]] = {
    "perplexity":   (2.0,  20.0),
    "entropy":      (0.5,   6.0),
    "ttr":          (0.2,   1.0),
    "word_length":  (2.0,  10.0),
    "hapax":        (0.1,   1.0),
    "char_entropy": (1.0,   5.0),
}


def analyse(
    source: Union[str, Path],
    *,
    metric: str = "perplexity",
    order: int = 9,
    n_octaves: int = 6,
    persistence: float = 0.55,
    detail_boost: float = 0.6,
    invert: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    size: int = 10,
    dpi: int = 200,
) -> plt.Figure:
    """
    One-call convenience: parse → multi-scale metric → Hilbert image → plot.

    Parameters
    ----------
    source : str or Path
        Raw text or a path to a .txt / .json / .csv file.
    metric : str
        Which metric to compute.  See ``perplexfern.AVAILABLE_METRICS``.
    order : int
        Hilbert-curve order (image = 2^order × 2^order).  Default 9 → 512².
    n_octaves : int
        Number of scale layers to blend (more = finer detail).
    persistence : float
        Amplitude decay per octave (0-1).
    detail_boost : float
        Laplacian edge-enhancement strength (0 = off).
    invert : bool
        Swap black / white.
    save_path : str or Path, optional
        Save the figure to this path.
    title : str, optional
        Title overlaid on the image.
    size, dpi : int
        Figure size and resolution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    text = parse(source)

    metric_fn = AVAILABLE_METRICS.get(metric)
    if metric_fn is None:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Available: {', '.join(sorted(AVAILABLE_METRICS))}"
        )

    metric_range = _METRIC_RANGES.get(metric, (2.0, 20.0))

    # Compute a coarse windowed signal for the sparkline + global value
    coarse_signal = compute_windowed(text, metric=metric)
    global_value = float(np.mean(coarse_signal))

    image = multi_scale_image(
        text,
        metric_fn,
        order=order,
        n_octaves=n_octaves,
        persistence=persistence,
        detail_boost=detail_boost,
        metric_range=metric_range,
    )

    fig = render(
        image,
        save_path=save_path,
        title=title,
        metric_label=metric,
        metric_value=global_value,
        metric_signal=coarse_signal,
        invert=invert,
        size=size,
        dpi=dpi,
    )
    return fig


def analyse_metrics(source: Union[str, Path]) -> TextMetrics:
    """Parse *source* and return summary metrics (no plot)."""
    return compute_summary(parse(source))


def analyse_galaxy(
    source: Union[str, Path],
    *,
    color_metric: str = "perplexity",
    size_metric: str = "entropy",
    resolution: int = 2048,
    point_size: float = 1.0,
    palette: str = "galaxy",
    glow: float = 6.0,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    size: int = 12,
    dpi: int = 200,
) -> plt.Figure:
    """
    One-call convenience: parse → multi-metric windows → PCA scatter → plot.

    Produces a colour-splashed point-cloud image (dark background, glowing
    clusters) inspired by the Open Syllabus Galaxy.

    Parameters
    ----------
    source : str or Path
        Raw text or a path to a .txt / .json / .csv file.
    color_metric : str
        Metric that determines each point's colour.
    size_metric : str
        Metric that determines each point's radius.
    resolution : int
        Canvas side length in pixels (default 2048).
    point_size : float
        Base point radius multiplier.
    palette : str
        ``"galaxy"`` for the built-in neon palette, or any matplotlib
        colormap name (e.g. ``"plasma"``, ``"viridis"``).
    glow : float
        Gaussian bloom radius (0 = off).
    save_path : str or Path, optional
        Save the figure here.
    title : str, optional
        Title overlaid on the image.
    size, dpi : int
        Figure size and resolution.

    Returns
    -------
    matplotlib.figure.Figure
    """
    text = parse(source)

    # Use all available metrics for the scatter embedding
    metric_dict = {
        name: fn
        for name, fn in sorted(AVAILABLE_METRICS.items())
    }

    positions, features = galaxy_points(text, metric_dict)
    metric_names = list(metric_dict.keys())

    image = galaxy_image(
        positions, features, metric_names,
        color_metric=color_metric,
        size_metric=size_metric,
        resolution=resolution,
        point_size=point_size,
        palette=palette,
        glow_sigma=glow,
    )

    # Compute a coarse signal for the sparkline
    coarse_signal = compute_windowed(text, metric=color_metric)
    global_value = float(np.mean(coarse_signal))

    fig = render_galaxy(
        image,
        save_path=save_path,
        title=title,
        metric_label=color_metric,
        metric_value=global_value,
        metric_signal=coarse_signal,
        size=size,
        dpi=dpi,
    )
    return fig


__all__ = [
    "analyse",
    "analyse_galaxy",
    "analyse_metrics",
    "parse",
    "compute_summary",
    "compute_windowed",
    "signal_to_image",
    "multi_scale_image",
    "galaxy_points",
    "galaxy_image",
    "render",
    "render_to_bytes",
    "render_galaxy",
    "render_galaxy_to_bytes",
    "TextMetrics",
    "AVAILABLE_METRICS",
]
