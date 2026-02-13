"""
Galaxy visualization – renders a colour point-cloud as a matplotlib figure.

This module mirrors ``visualize.py`` (which handles the B&W Hilbert fractal)
but produces a dark-background, colour-splashed scatter image inspired by
the Open Syllabus Galaxy.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_galaxy(
    image: np.ndarray,
    *,
    size: int = 12,
    dpi: int = 200,
    save_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    metric_label: Optional[str] = None,
    metric_value: Optional[float] = None,
    metric_signal: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Render an RGBA galaxy image as a matplotlib figure.

    Parameters
    ----------
    image : np.ndarray
        (H, W, 4) RGBA float32 array in [0, 1].
    size : int
        Figure side length in inches.
    dpi : int
        Resolution.
    save_path : str or Path, optional
        Write the figure here.
    title : str, optional
        Title overlaid on the image.
    metric_label : str, optional
        Small label in the bottom-right corner.
    metric_value : float, optional
        Global metric value to display.
    metric_signal : np.ndarray, optional
        1-D windowed metric values for an optional sparkline.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)
    bg = tuple(float(image[0, 0, ch]) for ch in range(3))
    bg_hex = "#{:02x}{:02x}{:02x}".format(
        int(bg[0] * 255), int(bg[1] * 255), int(bg[2] * 255),
    )
    fig.patch.set_facecolor(bg_hex)
    ax.set_facecolor(bg_hex)

    ax.imshow(image, interpolation="bilinear")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])

    if title:
        ax.text(
            0.5, 0.97, title,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold",
            ha="center", va="top",
            color="white", alpha=0.85,
            fontfamily="monospace",
        )

    # ── corner info ─────────────────────────────────────────────────────
    _draw_corner(ax, metric_label, metric_value, metric_signal)

    if save_path is not None:
        fig.savefig(
            str(save_path), dpi=dpi,
            bbox_inches="tight", pad_inches=0,
            facecolor=bg_hex,
        )

    return fig


def render_galaxy_to_bytes(
    image: np.ndarray,
    *,
    fmt: str = "png",
    **kwargs,
) -> bytes:
    """Return raw image bytes; the figure is closed after export."""
    fig = render_galaxy(image, **kwargs)
    bg_hex = fig.get_facecolor()
    buf = io.BytesIO()
    fig.savefig(
        buf, format=fmt,
        bbox_inches="tight", pad_inches=0,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── internal ────────────────────────────────────────────────────────────────

def _draw_corner(ax, label, value, signal) -> None:
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

    if signal is not None and len(signal) > 2:
        from perplexfern.visualize import _kde_smooth
        spark_ax = ax.inset_axes(
            [0.78, 0.035, 0.19, 0.06],
            transform=ax.transAxes,
        )
        spark_ax.set_facecolor("none")
        for spine in spark_ax.spines.values():
            spine.set_visible(False)
        spark_ax.tick_params(
            left=False, bottom=False,
            labelleft=False, labelbottom=False,
        )
        xs, ys = _kde_smooth(signal, n_points=200)
        ys = ys / (ys.max() + 1e-9)
        spark_ax.fill_between(xs, 0, ys, color="white", alpha=0.35, linewidth=0)
        spark_ax.plot(xs, ys, color="white", alpha=0.6, linewidth=0.9)
        spark_ax.axvline(
            float(np.mean(signal)),
            color="white", alpha=0.8, linewidth=0.7, linestyle="--",
        )
        spark_ax.set_xlim(xs[0], xs[-1])
        spark_ax.set_ylim(0, 1.15)
