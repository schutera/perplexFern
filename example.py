"""Generate example fractal and galaxy images for comparison."""
import perplexfern

# ── Example 1: Dickens – repetitive literary prose ──────────────────────────
dickens = (
    "It was the best of times, it was the worst of times, "
    "it was the age of wisdom, it was the age of foolishness, "
    "it was the epoch of belief, it was the epoch of incredulity, "
    "it was the season of Light, it was the season of Darkness, "
    "it was the spring of hope, it was the winter of despair, "
    "we had everything before us, we had nothing before us, "
    "we were all going direct to Heaven, we were all going direct the other way. "
) * 12

# ── Example 2: Biology textbook – dense technical vocabulary ────────────────
biology = (
    "The mitochondria is the powerhouse of the cell. "
    "Adenosine triphosphate, commonly abbreviated as ATP, serves as the "
    "primary energy currency in biological systems. Oxidative phosphorylation "
    "occurs within the inner mitochondrial membrane, coupling electron "
    "transport with chemiosmotic gradient formation to synthesize ATP from "
    "adenosine diphosphate and inorganic phosphate. Glycolysis, the Krebs "
    "cycle, and the electron transport chain constitute the three major "
    "metabolic pathways of cellular respiration. The citric acid cycle "
    "generates reduced coenzymes NADH and FADH2, which donate electrons to "
    "the respiratory chain complexes embedded in the cristae. Proton motive "
    "force across the membrane drives ATP synthase, a molecular rotary motor "
    "that catalyzes the phosphorylation of ADP. Substrate-level "
    "phosphorylation also contributes to net ATP yield during glycolysis "
    "and the citric acid cycle, though oxidative phosphorylation produces "
    "the vast majority of cellular ATP under aerobic conditions. "
) * 8

if __name__ == "__main__":
    for label, text in [("Dickens", dickens), ("Biology", biology)]:
        m = perplexfern.analyse_metrics(text)
        print(f"=== {label} ===")
        print(m.summary())
        print()

    # ── B&W Hilbert fractals ────────────────────────────────────────────
    perplexfern.analyse(
        dickens,
        metric="perplexity",
        title="Dickens – Perplexity",
        save_path="fractal_dickens.png",
    )
    perplexfern.analyse(
        biology,
        metric="perplexity",
        title="Biology – Perplexity",
        save_path="fractal_biology.png",
    )
    print("Saved fractal_dickens.png and fractal_biology.png")

    # ── Galaxy point-cloud scatter ──────────────────────────────────────
    perplexfern.analyse_galaxy(
        dickens,
        title="Dickens – Galaxy",
        save_path="galaxy_dickens.png",
    )
    perplexfern.analyse_galaxy(
        biology,
        title="Biology – Galaxy",
        save_path="galaxy_biology.png",
    )
    print("Saved galaxy_dickens.png and galaxy_biology.png")
