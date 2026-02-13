"""
Fractal module – maps text-metric signals onto a 2-D square image
using a **Hilbert curve** with **multi-scale octave blending**.

How it works
------------
1.  The metric (e.g. perplexity) is computed at *many* window sizes
    (octaves), from coarse to fine.
2.  Each 1-D signal is resampled and laid onto a Hilbert curve to
    produce a 2-D layer.
3.  Layers are weighted and summed exactly like fractal Brownian
    motion (large scales dominate, fine scales add detail).
4.  Adaptive histogram equalization spreads the tonal range so
    subtle structure becomes visible.
5.  Optional Laplacian-based detail enhancement makes edges pop.

The result: a genuinely intricate, data-driven fractal.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ── Hilbert curve ───────────────────────────────────────────────────────────

def _d2xy(n: int, d: int) -> tuple[int, int]:
    """Convert Hilbert-curve index *d* to (x, y) in an n×n grid."""
    x = y = 0
    s = 1
    while s < n:
        rx = 1 if (d & 2) else 0
        ry = 1 if (d & 1) ^ rx else 0
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        d >>= 2
        s <<= 1
    return x, y


def _build_hilbert_lut(order: int) -> np.ndarray:
    side = 1 << order
    n = side * side
    lut = np.empty((n, 2), dtype=np.int32)
    for d in range(n):
        x, y = _d2xy(side, d)
        lut[d] = (y, x)          # row = y, col = x
    return lut


# ── LUT cache (order → lut) so we don't rebuild every call ─────────────────

_LUT_CACHE: dict[int, np.ndarray] = {}


def _get_lut(order: int) -> np.ndarray:
    if order not in _LUT_CACHE:
        _LUT_CACHE[order] = _build_hilbert_lut(order)
    return _LUT_CACHE[order]


# ── single-signal → image ──────────────────────────────────────────────────

def _signal_to_layer(signal: np.ndarray, order: int) -> np.ndarray:
    """Map a 1-D signal onto a 2-D image via Hilbert curve (one layer)."""
    side = 1 << order
    n_pixels = side * side
    resampled = _resample(signal, n_pixels)

    lo, hi = resampled.min(), resampled.max()
    if hi - lo > 1e-12:
        normed = (resampled - lo) / (hi - lo)
    else:
        normed = np.full_like(resampled, 0.5)

    lut = _get_lut(order)
    image = np.zeros((side, side), dtype=np.float64)
    for d in range(n_pixels):
        r, c = lut[d]
        image[r, c] = normed[d]
    return image


# ── multi-scale octave blending ────────────────────────────────────────────

def multi_scale_image(
    text: str,
    metric_fn: Callable[[str], float],
    *,
    order: int = 9,
    n_octaves: int = 6,
    lacunarity: float = 2.0,
    persistence: float = 0.55,
    detail_boost: float = 0.6,
    metric_range: tuple[float, float] = (2.0, 20.0),
) -> np.ndarray:
    """
    Produce a square fractal image by blending *n_octaves* of a
    windowed text metric.

    The **absolute level** of the metric controls the visual character:

    - Low metric  → sparse, bright, smooth, large open regions
    - High metric → dense, dark, intricate, fine-grained texture

    This means two different texts will produce images that look
    obviously different at first glance.

    Parameters
    ----------
    text : str
        The raw input text.
    metric_fn : callable(str) → float
        A function that scores a text chunk (e.g. perplexity of a window).
    order : int
        Hilbert-curve order.  Image will be 2^order × 2^order
        (default 9 → 512 × 512).
    n_octaves : int
        Number of scale layers to blend (default 6).
    lacunarity : float
        How fast the window size shrinks per octave (default 2.0).
    persistence : float
        Amplitude decay per octave (default 0.55).
    detail_boost : float
        Strength of the Laplacian detail-enhancement pass (0 = off).
    metric_range : (float, float)
        Expected (low, high) range of the metric.  Used to anchor the
        global intensity so that absolute differences are visible.
        Default (2, 20) works well for perplexity.

    Returns
    -------
    np.ndarray   shape (side, side), values in [0, 1]
    """
    import math

    n = len(text)
    side = 1 << order

    # ── collect octave layers AND raw metric values ─────────────────────
    base_window = max(20, n // 8)
    layers: list[tuple[float, np.ndarray]] = []    # (amplitude, layer)
    all_raw_values: list[float] = []               # raw metric values
    amplitude = 1.0

    for octave in range(n_octaves):
        window = max(8, int(base_window / (lacunarity ** octave)))
        stride = max(1, window // 3)
        signal = _windowed_signal(text, metric_fn, window, stride)

        if len(signal) < 2:
            continue

        all_raw_values.extend(signal.tolist())
        layer = _signal_to_layer(signal, order)
        layers.append((amplitude, layer))
        amplitude *= persistence

    if not layers:
        return np.full((side, side), 0.5)

    # ── compute global intensity anchor from raw values ─────────────────
    raw_arr = np.asarray(all_raw_values)
    global_mean = float(np.mean(raw_arr))
    global_std = float(np.std(raw_arr))

    # Normalise global_mean into [0, 1] using expected metric range
    m_lo, m_hi = metric_range
    intensity = np.clip((global_mean - m_lo) / (m_hi - m_lo), 0.0, 1.0)

    # Also measure how variable the signal is (coefficient of variation)
    cv = global_std / max(global_mean, 1e-6)
    variability = np.clip(cv, 0.0, 1.0)

    # ── blend octaves with intensity-modulated persistence ──────────────
    # Higher intensity → more fine-detail weight
    eff_persist_boost = 0.3 * intensity   # up to +0.3 extra persistence
    composite = np.zeros((side, side), dtype=np.float64)
    total_amp = 0.0
    amp = 1.0
    eff_persistence = persistence + eff_persist_boost

    for i, (_, layer) in enumerate(layers):
        composite += amp * layer
        total_amp += amp
        amp *= eff_persistence

    if total_amp > 0:
        composite /= total_amp

    # ── adaptive histogram equalization ─────────────────────────────────
    # Lower intensity → more aggressive clipping (flattens detail → simpler)
    clip = 1.5 + 2.0 * (1.0 - intensity)
    composite = _adaptive_histeq(composite, clip_limit=clip, n_tiles=8)

    # ── detail enhancement ──────────────────────────────────────────────
    eff_detail = detail_boost * (0.4 + 0.8 * intensity)
    if eff_detail > 0:
        composite = _laplacian_sharpen(composite, strength=eff_detail)

    # ── normalise to [0, 1] ─────────────────────────────────────────────
    lo, hi = composite.min(), composite.max()
    if hi - lo > 1e-12:
        composite = (composite - lo) / (hi - lo)
    else:
        composite[:] = 0.5

    # ── intensity-driven tone curve ─────────────────────────────────────
    # This is the key: the absolute metric level shifts the entire
    # brightness & density of the image.
    #
    #   low  intensity → high gamma → pushes pixels toward white
    #                     (sparse, airy, lots of white space)
    #   high intensity → low gamma  → pushes pixels toward black
    #                     (dense, dark, intricate)
    #
    # We also apply a sigmoid to boost contrast, with steepness
    # modulated by the signal variability.

    # Gamma: ranges from ~2.2 (bright) at intensity=0 to ~0.45 (dark) at 1
    gamma = 2.2 - 1.75 * intensity

    # Sigmoid contrast: steeper when the text is more variable
    sigmoid_k = 6.0 + 6.0 * variability     # steepness 6–12
    sigmoid_mid = 0.5 - 0.15 * intensity     # shift midpoint darker

    composite = np.power(np.clip(composite, 1e-9, 1.0), gamma)
    composite = _sigmoid(composite, k=sigmoid_k, mid=sigmoid_mid)

    # ── final clamp ─────────────────────────────────────────────────────
    lo, hi = composite.min(), composite.max()
    if hi - lo > 1e-12:
        composite = (composite - lo) / (hi - lo)
    else:
        composite[:] = 0.5

    return composite


# ── simple (non-octave) entry-point kept for backward compat ────────────────

def signal_to_image(
    signal: np.ndarray,
    order: int | None = None,
) -> np.ndarray:
    """Map a single 1-D signal to a 2-D image (no octave blending)."""
    n_vals = len(signal)
    if order is None:
        order = 1
        while (1 << order) ** 2 < n_vals:
            order += 1
    return _signal_to_layer(signal, order)


# ── internal helpers ────────────────────────────────────────────────────────

def _windowed_signal(
    text: str,
    fn: Callable[[str], float],
    window: int,
    stride: int,
) -> np.ndarray:
    """Slide *fn* over *text* and return the 1-D value array."""
    values: list[float] = []
    start = 0
    while start + window <= len(text):
        values.append(fn(text[start : start + window]))
        start += stride
    if not values:
        values.append(fn(text))
    return np.asarray(values, dtype=np.float64)


def _resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) == target_len:
        return arr.copy()
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, arr)


def _adaptive_histeq(
    img: np.ndarray,
    clip_limit: float = 2.5,
    n_tiles: int = 8,
) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalization (CLAHE),
    implemented in pure NumPy (no OpenCV dependency).
    """
    h, w = img.shape

    # Quantise to 256 levels for histogram operations
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-12:
        return img.copy()
    scaled = ((img - lo) / (hi - lo) * 255).astype(np.int32)

    tile_h = max(1, h // n_tiles)
    tile_w = max(1, w // n_tiles)
    out = np.zeros_like(img)

    for ty in range(0, h, tile_h):
        for tx in range(0, w, tile_w):
            by, bx = min(ty + tile_h, h), min(tx + tile_w, w)
            tile = scaled[ty:by, tx:bx]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)

            # Clip histogram
            excess = 0.0
            limit = clip_limit * tile.size / 256
            for i in range(256):
                if hist[i] > limit:
                    excess += hist[i] - limit
                    hist[i] = limit
            hist += excess / 256

            # CDF
            cdf = hist.cumsum()
            cdf_min = cdf[cdf > 0].min() if np.any(cdf > 0) else 0
            denom = tile.size - cdf_min
            if denom > 0:
                cdf = (cdf - cdf_min) / denom
            cdf = np.clip(cdf, 0, 1)

            # Map
            mapped = cdf[tile]
            out[ty:by, tx:bx] = mapped

    return out


def _laplacian_sharpen(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Add a scaled Laplacian (edge detail) back into the image."""
    # 3×3 Laplacian kernel via convolution
    padded = np.pad(img, 1, mode="reflect")
    lap = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] +
        padded[1:-1, :-2] + padded[1:-1, 2:] -
        4 * padded[1:-1, 1:-1]
    )
    return img - strength * lap


def _sigmoid(arr: np.ndarray, k: float = 8.0, mid: float = 0.5) -> np.ndarray:
    """Apply a sigmoid contrast curve:  1 / (1 + exp(-k*(x - mid)))."""
    return 1.0 / (1.0 + np.exp(-k * (arr - mid)))
