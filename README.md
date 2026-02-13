# perplexFern 🌿

Visualise the complexity of natural text as a fractal fern.

Feed in **any text** — a raw string, a `.txt` file, a `.json` document, or a `.csv` spreadsheet — and perplexFern will:

1. **Compute** perplexity, Shannon entropy, burstiness, vocabulary richness, and more.
2. **Generate** a unique Barnsley-fern fractal whose shape is warped by those metrics.
3. **Return** a square, black-and-white plot with an optional metrics overlay.

Every text produces a *different* fern. Repetitive text yields tight, regular fronds; rich, varied prose fans out into elaborate shapes.

---

## Installation

```bash
pip install -e .
```

## Quick start

```python
import perplexfern

# From a raw string
fig = perplexfern.analyse(
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs.",
    save_path="fern.png",
)

# From a file
fig = perplexfern.analyse("path/to/essay.txt", save_path="essay_fern.png")
```

## Metrics only (no plot)

```python
m = perplexfern.analyse_metrics("Some interesting paragraph...")
print(m.summary())
```

## Step-by-step API

```python
from perplexfern import parse, compute_metrics, generate_points, render

text    = parse("data.csv")
metrics = compute_metrics(text)
points  = generate_points(metrics, n_points=300_000, seed=42)
fig     = render(points, metrics, show_metrics=True)
fig.savefig("output.png", dpi=200)
```

## Metrics computed

| Metric | Description |
|---|---|
| **Perplexity** | Character-level bigram perplexity (Laplace-smoothed) |
| **Entropy** | Shannon entropy over the word distribution (bits) |
| **Cross-entropy** | Character-level bigram cross-entropy |
| **Type-token ratio** | Unique words / total words |
| **Hapax ratio** | Words appearing exactly once / total words |
| **Yule's K** | Vocabulary richness (lower = richer) |
| **Burstiness** | (σ − μ) / (σ + μ) of inter-word gaps |
| **Mean / Std word length** | Average and standard deviation of word lengths |

## How the fractal works

The classic **Barnsley fern** is an Iterated Function System (IFS) with four affine transformations, each selected with a fixed probability. perplexFern **warps** those coefficients using the normalised text metrics:

- **Perplexity** stretches the main body and stem.
- **Entropy** shifts the vertical offset.
- **Burstiness** skews the leaflet angles.
- **Vocabulary richness** (TTR, hapax, Yule's K) reshapes the side leaflets.

The result is a deterministic mapping: same text → same fern (given the same seed).

## License

MIT
