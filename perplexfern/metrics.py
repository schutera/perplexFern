"""
Metrics module – computes text metrics over sliding windows.

Each metric function takes raw text and returns a 1-D NumPy array of
per-window values.  The windowed signal is what gets mapped onto the
fractal image.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ── tokeniser ───────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-zA-Z\u00C0-\u024F]+(?:['\u2019][a-zA-Z]+)*")


def _tokenise(text: str) -> list[str]:
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


# ── windowed metric computation ─────────────────────────────────────────────

AVAILABLE_METRICS: dict[str, Callable] = {}   # filled by @_register


def _register(name: str):
    """Decorator that registers a windowed-metric function."""
    def decorator(fn):
        AVAILABLE_METRICS[name] = fn
        return fn
    return decorator


def compute_windowed(
    text: str,
    metric: str = "perplexity",
    window_chars: int | None = None,
    stride_chars: int | None = None,
) -> np.ndarray:
    """
    Compute *metric* over a sliding character window.

    Parameters
    ----------
    text : str
        Input text.
    metric : str
        One of the keys in ``AVAILABLE_METRICS``.
    window_chars : int, optional
        Window width in characters.  Default: ``len(text) // 64`` (clamped ≥ 20).
    stride_chars : int, optional
        Step size in characters.  Default: ``window_chars // 2``.

    Returns
    -------
    np.ndarray
        1-D float array, one value per window position.
    """
    if not text or not text.strip():
        raise ValueError("Input text must be non-empty.")

    fn = AVAILABLE_METRICS.get(metric)
    if fn is None:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Available: {', '.join(sorted(AVAILABLE_METRICS))}"
        )

    n = len(text)
    if window_chars is None:
        window_chars = max(20, n // 64)
    if stride_chars is None:
        stride_chars = max(1, window_chars // 2)

    values: list[float] = []
    start = 0
    while start + window_chars <= n:
        chunk = text[start : start + window_chars]
        values.append(fn(chunk))
        start += stride_chars

    # If text is too short to give even one window, just do the whole text
    if not values:
        values.append(fn(text))

    return np.array(values, dtype=np.float64)


# ── individual metric functions (operate on a single chunk) ─────────────────

@_register("perplexity")
def _perplexity(chunk: str) -> float:
    """Character-level bigram perplexity (Laplace-smoothed)."""
    ce = _char_bigram_cross_entropy(chunk)
    return 2.0 ** ce


@_register("entropy")
def _entropy(chunk: str) -> float:
    """Shannon entropy over the word distribution (bits)."""
    words = _tokenise(chunk)
    if not words:
        return 0.0
    freq = Counter(words)
    n = len(words)
    return -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)


@_register("ttr")
def _ttr(chunk: str) -> float:
    """Type-token ratio (unique words / total words)."""
    words = _tokenise(chunk)
    if not words:
        return 0.0
    return len(set(words)) / len(words)


@_register("word_length")
def _word_length(chunk: str) -> float:
    """Mean word length."""
    words = _tokenise(chunk)
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


@_register("hapax")
def _hapax(chunk: str) -> float:
    """Hapax ratio – proportion of words that appear exactly once."""
    words = _tokenise(chunk)
    if not words:
        return 0.0
    freq = Counter(words)
    return sum(1 for c in freq.values() if c == 1) / len(words)


@_register("char_entropy")
def _char_entropy(chunk: str) -> float:
    """Shannon entropy over individual characters (bits)."""
    if not chunk:
        return 0.0
    freq = Counter(chunk.lower())
    n = len(chunk)
    return -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)


# ── helper ──────────────────────────────────────────────────────────────────

def _char_bigram_cross_entropy(text: str) -> float:
    t = text.lower()
    if len(t) < 2:
        return 0.0
    vocab = sorted(set(t))
    vs = len(vocab)
    if vs <= 1:
        return 0.0

    bigram_counts: Counter = Counter()
    unigram_counts: Counter = Counter()
    for i in range(len(t) - 1):
        bigram_counts[(t[i], t[i + 1])] += 1
        unigram_counts[t[i]] += 1
    unigram_counts[t[-1]] += 1

    total = len(t) - 1
    ce = 0.0
    for (c1, _), count in bigram_counts.items():
        p = (count + 1) / (unigram_counts[c1] + vs)
        ce -= count * math.log2(p)
    return ce / total


# ── convenience: global summary ─────────────────────────────────────────────

@dataclass(frozen=True)
class TextMetrics:
    """Summary metrics for an entire text."""
    perplexity: float
    entropy: float
    ttr: float
    hapax: float
    char_entropy: float
    mean_word_length: float
    n_chars: int
    n_words: int
    n_unique_words: int

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name)
                for f in self.__dataclass_fields__.values()}

    def summary(self) -> str:
        lines = ["── Text Metrics ──"]
        for k, v in self.as_dict().items():
            if isinstance(v, float):
                lines.append(f"  {k:>20s}: {v:.4f}")
            else:
                lines.append(f"  {k:>20s}: {v}")
        return "\n".join(lines)


def compute_summary(text: str) -> TextMetrics:
    """Compute a single set of summary metrics for the entire text."""
    if not text or not text.strip():
        raise ValueError("Input text must be non-empty.")
    words = _tokenise(text)
    freq = Counter(words)
    n = len(words)
    nu = len(freq)
    return TextMetrics(
        perplexity=_perplexity(text),
        entropy=_entropy(text),
        ttr=nu / n if n else 0.0,
        hapax=sum(1 for c in freq.values() if c == 1) / n if n else 0.0,
        char_entropy=_char_entropy(text),
        mean_word_length=sum(len(w) for w in words) / n if n else 0.0,
        n_chars=len(text),
        n_words=n,
        n_unique_words=nu,
    )
