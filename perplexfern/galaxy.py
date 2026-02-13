"""
Galaxy module – colour-splashed point-cloud visualization inspired by
the Open Syllabus Galaxy (galaxy.opensyllabus.org).

The idea
--------
Instead of mapping the metric signal onto a Hilbert-curve grid, we
scatter *points* in 2-D space whose positions, colours, and sizes are
all driven by the windowed text metrics.

1.  **Multiple metrics** are computed simultaneously over sliding windows
    to yield a small feature vector per window.
2.  A lightweight, pure-NumPy **t-SNE-like projection** (or a simpler
    spectral / PCA projection) embeds these vectors in 2-D so that
    similarly-scored windows cluster together.
3.  Each point is coloured by its primary metric value using a
    perceptual colour-map, sized by a secondary metric, and drawn with
    additive-style alpha blending on a dark background – producing the
    characteristic glowing-cluster look of the OS Galaxy.

Dependencies: only NumPy + matplotlib (no sklearn / umap / scipy).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import numpy as np


# ── Public API ──────────────────────────────────────────────────────────────

def galaxy_points(
    text: str,
    metrics: dict[str, Callable[[str], float]],
    *,
    window_chars: int | None = None,
    stride_chars: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute multi-metric feature vectors over sliding windows.

    Parameters
    ----------
    text : str
        The input text.
    metrics : dict[str, callable]
        Mapping of metric name → function(chunk) → float.
    window_chars, stride_chars : int, optional
        Window / stride width.  Defaults mirror ``metrics.compute_windowed``.

    Returns
    -------
    positions : ndarray, shape (N, 2)
        2-D coordinates for each window (PCA projection).
    features : ndarray, shape (N, M)
        Raw metric values per window (columns ordered as ``metrics`` keys).
    """
    n = len(text)
    if window_chars is None:
        window_chars = max(20, n // 64)
    if stride_chars is None:
        stride_chars = max(1, window_chars // 2)

    metric_fns = list(metrics.values())
    m = len(metric_fns)

    # ── windowed feature matrix ─────────────────────────────────────────
    rows: list[list[float]] = []
    start = 0
    while start + window_chars <= n:
        chunk = text[start: start + window_chars]
        rows.append([fn(chunk) for fn in metric_fns])
        start += stride_chars

    if not rows:
        rows.append([fn(text) for fn in metric_fns])

    features = np.asarray(rows, dtype=np.float64)        # (N, M)

    # ── project to 2-D via PCA ──────────────────────────────────────────
    positions = _pca_2d(features)

    return positions, features


def galaxy_image(
    positions: np.ndarray,
    features: np.ndarray,
    metric_names: Sequence[str],
    *,
    color_metric: str | None = None,
    size_metric: str | None = None,
    resolution: int = 2048,
    point_size: float = 1.0,
    palette: str = "galaxy",
    background: tuple[float, float, float] = (0.02, 0.02, 0.06),
    glow_sigma: float = 0.0,
) -> np.ndarray:
    """
    Render a colour point-cloud image from projected positions + features.

    Parameters
    ----------
    positions : ndarray (N, 2)
        2-D coordinates.
    features : ndarray (N, M)
        Raw metric values (same column order as *metric_names*).
    metric_names : list[str]
        Column labels for *features*.
    color_metric : str, optional
        Metric used to assign point colour.  Default: first metric.
    size_metric : str, optional
        Metric used to scale point size.  Default: second metric (or first).
    resolution : int
        Output image side length in pixels.
    point_size : float
        Base radius of each point (pixels).  Scales with resolution.
    palette : str
        ``"galaxy"`` (custom neon palette) or any matplotlib colormap name.
    background : (r, g, b) floats 0-1
        Background colour.
    glow_sigma : float
        If > 0, apply a Gaussian glow/bloom pass after rendering.

    Returns
    -------
    ndarray, shape (resolution, resolution, 4)   – RGBA float32 in [0, 1]
    """
    n = len(positions)
    names = list(metric_names)

    # ── determine colour & size columns ─────────────────────────────────
    ci = names.index(color_metric) if color_metric and color_metric in names else 0
    si = (
        names.index(size_metric) if size_metric and size_metric in names
        else min(1, len(names) - 1)
    )

    color_vals = _normalise(features[:, ci])
    size_vals = _normalise(features[:, si])

    # ── map positions to pixel coordinates ──────────────────────────────
    margin = 0.06
    px, py = _to_pixel(positions, resolution, margin)

    # ── assign colours via palette ──────────────────────────────────────
    rgba = _map_palette(color_vals, palette)                # (N, 4)

    # ── assign radii ────────────────────────────────────────────────────
    base_r = point_size * (resolution / 512)
    radii = base_r * (0.5 + 2.0 * size_vals)               # 0.5–2.5 × base

    # ── render to canvas ────────────────────────────────────────────────
    canvas = _render_points(
        px, py, rgba, radii,
        resolution=resolution,
        background=background,
    )

    # ── optional glow ───────────────────────────────────────────────────
    if glow_sigma > 0:
        canvas = _glow(canvas, sigma=glow_sigma, background=background)

    return canvas


# ── PCA projection (pure NumPy) ─────────────────────────────────────────────

def _pca_2d(X: np.ndarray) -> np.ndarray:
    """Project rows of X into 2-D via the first two principal components."""
    X = X.astype(np.float64)
    n, m = X.shape

    if m <= 2:
        out = np.zeros((n, 2), dtype=np.float64)
        out[:, :m] = X
        return out

    # centre
    mean = X.mean(axis=0)
    Xc = X - mean

    # covariance
    cov = (Xc.T @ Xc) / max(n - 1, 1)

    # eigen decomposition (symmetric → use eigh)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # top-2 components (eigh returns ascending order)
    idx = np.argsort(eigvals)[::-1][:2]
    W = eigvecs[:, idx]                          # (m, 2)

    projected = Xc @ W                           # (n, 2)

    # Add a small amount of jitter so perfectly-equal windows don't stack
    rng = np.random.default_rng(42)
    span = projected.max() - projected.min() + 1e-9
    jitter = rng.normal(0, span * 0.005, projected.shape)
    projected += jitter

    return projected


# ── Colour palettes ─────────────────────────────────────────────────────────

_GALAXY_STOPS = np.array([
    # t,     R,    G,    B
    [0.00, 0.10, 0.15, 0.60],   # deep blue
    [0.18, 0.25, 0.05, 0.70],   # violet
    [0.35, 0.75, 0.10, 0.55],   # magenta / pink
    [0.50, 0.95, 0.35, 0.15],   # orange
    [0.65, 1.00, 0.85, 0.10],   # yellow
    [0.80, 0.20, 0.90, 0.40],   # green
    [0.92, 0.15, 0.70, 0.95],   # cyan
    [1.00, 0.55, 0.45, 1.00],   # lilac
], dtype=np.float64)


def _map_palette(values: np.ndarray, palette: str) -> np.ndarray:
    """Map normalised [0, 1] values → RGBA (N, 4)."""
    n = len(values)
    rgba = np.ones((n, 4), dtype=np.float32)

    if palette == "galaxy":
        stops = _GALAXY_STOPS
        ts = stops[:, 0]
        for ch in range(3):
            rgba[:, ch] = np.interp(values, ts, stops[:, ch + 1]).astype(np.float32)
        # Alpha: brighter in the middle of the range
        rgba[:, 3] = (0.35 + 0.55 * np.sqrt(values)).astype(np.float32)
    else:
        # Fall back to a matplotlib colormap
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap(palette)
        mapped = cmap(values)                                 # (N, 4)
        rgba[:, :] = mapped.astype(np.float32)
        rgba[:, 3] = (0.35 + 0.55 * np.sqrt(values)).astype(np.float32)

    return rgba


# ── Rasteriser ──────────────────────────────────────────────────────────────

def _render_points(
    px: np.ndarray,
    py: np.ndarray,
    rgba: np.ndarray,
    radii: np.ndarray,
    *,
    resolution: int,
    background: tuple[float, float, float],
) -> np.ndarray:
    """
    Splat soft circles onto an RGBA canvas with additive colour blending.

    Each point is drawn as a Gaussian "splat" – a small soft disc whose
    colour is added to the canvas, producing the glowing cluster effect.
    """
    canvas = np.zeros((resolution, resolution, 4), dtype=np.float64)
    # Pre-fill background (RGB only, alpha = 1)
    for ch in range(3):
        canvas[:, :, ch] = background[ch]
    canvas[:, :, 3] = 1.0

    n = len(px)
    # Sort points by radius (large first → small on top)
    order = np.argsort(radii)[::-1]

    for i in order:
        x, y = float(px[i]), float(py[i])
        r = float(radii[i])
        colour = rgba[i]

        # Bounding box
        ir = int(math.ceil(r * 2.5))   # cover ~2.5 sigma
        x0 = max(0, int(x) - ir)
        x1 = min(resolution, int(x) + ir + 1)
        y0 = max(0, int(y) - ir)
        y1 = min(resolution, int(y) + ir + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        # Distance field for the bounding box
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        sigma2 = r * r + 1e-9
        intensity = np.exp(-d2 / (2 * sigma2))       # Gaussian splat

        # Additive blending of colour
        alpha = intensity * float(colour[3])
        for ch in range(3):
            canvas[y0:y1, x0:x1, ch] += alpha * float(colour[ch])

    # Tone-map: soft clamp to [0, 1] using tanh-like curve
    for ch in range(3):
        c = canvas[:, :, ch]
        # Subtract background before tone-mapping the additive glow
        bg = background[ch]
        added = np.maximum(c - bg, 0.0)
        # Apply soft clamp: value → bg + (1-bg) * tanh(added * k)
        k = 2.5   # controls how quickly the glow saturates
        canvas[:, :, ch] = bg + (1.0 - bg) * np.tanh(added * k)

    canvas[:, :, 3] = 1.0
    return canvas.astype(np.float32)


# ── Glow / bloom pass ──────────────────────────────────────────────────────

def _glow(
    img: np.ndarray,
    sigma: float = 6.0,
    strength: float = 0.5,
    background: tuple[float, float, float] = (0.02, 0.02, 0.06),
) -> np.ndarray:
    """Add a soft bloom halo around bright regions."""
    blurred = _gaussian_blur(img[:, :, :3], sigma)

    out = img.copy()
    for ch in range(3):
        bright = np.maximum(blurred[:, :, ch] - background[ch], 0.0)
        out[:, :, ch] = np.clip(
            img[:, :, ch] + strength * bright, 0, 1,
        )
    return out


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (pure NumPy)."""
    radius = int(math.ceil(sigma * 3))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    h, w, c = img.shape
    tmp = np.empty_like(img)
    out = np.empty_like(img)

    # Horizontal pass
    padded = np.pad(img, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
    for ch in range(c):
        for row in range(h):
            for j in range(w):
                tmp[row, j, ch] = np.dot(
                    padded[row, j: j + 2 * radius + 1, ch], kernel,
                )

    # Vertical pass
    padded = np.pad(tmp, ((radius, radius), (0, 0), (0, 0)), mode="reflect")
    for ch in range(c):
        for col in range(w):
            for i in range(h):
                out[i, col, ch] = np.dot(
                    padded[i: i + 2 * radius + 1, col, ch], kernel,
                )

    return out


# ── small helpers ───────────────────────────────────────────────────────────

def _normalise(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def _to_pixel(
    positions: np.ndarray,
    resolution: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale (N, 2) positions into pixel coords with a margin."""
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    span = hi - lo
    span[span < 1e-12] = 1.0

    normed = (positions - lo) / span                          # [0, 1]

    px_lo = margin * resolution
    px_hi = (1.0 - margin) * resolution
    px = normed[:, 0] * (px_hi - px_lo) + px_lo
    py = normed[:, 1] * (px_hi - px_lo) + px_lo

    return px, py
