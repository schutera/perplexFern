"""
Visualization module – renders the 2-D metric image as a square
black-and-white plot with high visual fidelity.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(
    image: np.ndarray,
    *,
    size: int = 10,
    dpi: int = 200,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    metric_label: Optional[str] = None,
    metric_value: Optional[float] = None,
    metric_signal: Optional[np.ndarray] = None,
    invert: bool = False,
) -> plt.Figure:
    """
    Render a 2-D metric array as a square black-and-white image.

    Parameters
    ----------
    image : np.ndarray
        2-D array with values in [0, 1].
    size : int
        Side length of the figure in inches.
    dpi : int
        Resolution.
    save_path : str or Path, optional
        Save the figure here.
    title : str, optional
        Title above the plot.
    metric_label : str, optional
        Small label in the bottom-right corner.
    metric_value : float, optional
        Global metric value to display.
    metric_signal : np.ndarray, optional
        1-D array of windowed metric values – used to draw a mini
        distribution sparkline in the corner.
    invert : bool
        If True, high metric → black; low → white.

    Returns
    -------
    matplotlib.figure.Figure
    """
    display = image if not invert else (1.0 - image)

    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.imshow(
        display,
        cmap="gray",
        vmin=0, vmax=1,
        interpolation="bicubic",
    )
    ax.set_aspect("equal")
    ax.axis("off")

    # Remove all padding so the image fills the square completely
    ax.set_position([0, 0, 1, 1])

    if title:
        ax.text(
            0.5, 0.97, title,
            transform=ax.transAxes,
            fontsize=13, fontweight="bold",
            ha="center", va="top",
            color="white", alpha=0.7,
            fontfamily="monospace",
        )

    # ── bottom-right corner: sparkline + label + value ──────────────────
    _draw_corner_info(
        ax, metric_label, metric_value, metric_signal,
    )

    if save_path is not None:
        fig.savefig(
            str(save_path), dpi=dpi,
            bbox_inches="tight", pad_inches=0,
            facecolor="black",
        )

    return fig


def _draw_corner_info(
    ax,
    label: Optional[str],
    value: Optional[float],
    signal: Optional[np.ndarray],
) -> None:
    """Draw the metric sparkline, label, and value in the bottom-right."""
    from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform

    has_spark = signal is not None and len(signal) > 2
    has_text = label is not None or value is not None
    if not has_spark and not has_text:
        return

    # ── text line ───────────────────────────────────────────────────────
    parts: list[str] = []
    if label:
        parts.append(label)
    if value is not None:
        parts.append(f"{value:.2f}")
    caption = "  ".join(parts)

    if caption:
        ax.text(
            0.98, 0.015, caption,
            transform=ax.transAxes,
            fontsize=7, fontfamily="monospace",
            ha="right", va="bottom",
            color="white", alpha=0.55,
        )

    # ── sparkline distribution plot ─────────────────────────────────────
    if has_spark:
        # Create a tiny inset axes in the bottom-right
        # Position: right-aligned, just above the text line
        spark_ax = ax.inset_axes(
            [0.78, 0.035, 0.19, 0.06],  # [x, y, width, height] in axes coords
            transform=ax.transAxes,
        )
        spark_ax.set_facecolor("none")
        for spine in spark_ax.spines.values():
            spine.set_visible(False)
        spark_ax.tick_params(
            left=False, bottom=False,
            labelleft=False, labelbottom=False,
        )

        # Smooth KDE curve
        xs, ys = _kde_smooth(signal, n_points=200)
        # Normalise to [0, 1]
        ys = ys / (ys.max() + 1e-9)

        spark_ax.fill_between(
            xs, 0, ys,
            color="white", alpha=0.45,
            linewidth=0,
        )
        spark_ax.plot(
            xs, ys,
            color="white", alpha=0.7,
            linewidth=0.9,
        )
        # Mark the mean with a thin vertical line
        mean_val = float(np.mean(signal))
        spark_ax.axvline(
            mean_val, color="white", alpha=0.8,
            linewidth=0.7, linestyle="--",
        )
        spark_ax.set_xlim(xs[0], xs[-1])
        spark_ax.set_ylim(0, 1.15)


def _kde_smooth(
    data: np.ndarray,
    n_points: int = 200,
    bandwidth: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian kernel density estimate (pure NumPy, no scipy needed).

    Returns (xs, density) – a smooth, rounded curve over *n_points*.
    """
    data = np.asarray(data, dtype=np.float64)
    n = len(data)

    # Silverman's rule of thumb, scaled down to preserve detail
    if bandwidth is None:
        std = float(np.std(data, ddof=1)) if n > 1 else 1.0
        iqr = float(np.percentile(data, 75) - np.percentile(data, 25))
        spread = min(std, iqr / 1.34) if iqr > 0 else std
        bandwidth = 0.45 * spread * n ** (-0.2)
        bandwidth = max(bandwidth, 1e-6)

    lo = data.min() - 3 * bandwidth
    hi = data.max() + 3 * bandwidth
    xs = np.linspace(lo, hi, n_points)

    # Vectorised Gaussian kernel evaluation
    # Shape: (n_points, n_data)
    diff = (xs[:, None] - data[None, :]) / bandwidth
    density = np.exp(-0.5 * diff * diff).sum(axis=1)
    density /= n * bandwidth * np.sqrt(2 * np.pi)

    return xs, density


def render_to_bytes(
    image: np.ndarray,
    *,
    fmt: str = "png",
    **kwargs,
) -> bytes:
    """Return raw image bytes.  The figure is closed after export."""
    fig = render(image, **kwargs)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", pad_inches=0,
                facecolor="black")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
