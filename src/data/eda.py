"""Exploratory data analysis: figures + written report.

Everything is regenerated from ``data/processed/`` so no figure in the report is
a one-off screenshot that nobody can reproduce.

Run from the project root::

    python -m src.data.eda
"""

from __future__ import annotations

import math
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # headless: no display needed

import matplotlib.pyplot as plt
import pandas as pd

from src.data.schema import LABEL_NAMES, SPLITS
from src.utils.paths import FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR, ensure_dirs

PALETTE = {
    "sadness": "#4C72B0",
    "joy": "#DD8452",
    "love": "#C44E52",
    "anger": "#937860",
    "fear": "#55A868",
    "surprise": "#8172B3",
}
DPI = 130


def load() -> dict[str, pd.DataFrame]:
    frames = {}
    for split in SPLITS:
        path = PROCESSED_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data.build` first."
            )
        frames[split] = pd.read_csv(path, encoding="utf-8")
    return frames


def save(fig, name: str) -> str:
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")
    return f"figures/{name}"


# --------------------------------------------------------------- univariate --
def fig_class_distribution(train: pd.DataFrame) -> str:
    counts = train["label_name"].value_counts().reindex(LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index, counts.values,
                  color=[PALETTE[c] for c in counts.index])
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value,
                f"{value:,}\n{100 * value / total:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Class distribution (train)")
    ax.set_ylabel("messages")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "fig01_class_distribution.png")


def fig_length_distribution(train: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(train["n_words"], bins=40, color="#4C72B0", edgecolor="white")
    axes[0].axvline(train["n_words"].median(), color="crimson", ls="--",
                    label=f"median {train['n_words'].median():.0f}")
    axes[0].set_title("Words per message")
    axes[0].set_xlabel("words")
    axes[0].legend()

    axes[1].hist(train["n_chars"], bins=40, color="#55A868", edgecolor="white")
    axes[1].axvline(train["n_chars"].median(), color="crimson", ls="--",
                    label=f"median {train['n_chars'].median():.0f}")
    axes[1].set_title("Characters per message")
    axes[1].set_xlabel("characters")
    axes[1].legend()

    for ax in axes:
        ax.set_ylabel("messages")
        ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "fig02_length_distribution.png")


def fig_top_tokens(train: pd.DataFrame, n: int = 25) -> str:
    counter = Counter()
    for text in train["text_clean"]:
        counter.update(str(text).split())
    common = counter.most_common(n)[::-1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh([w for w, _ in common], [c for _, c in common], color="#4C72B0")
    ax.set_title(f"{n} most frequent tokens (train)")
    ax.set_xlabel("occurrences")
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "fig03_top_tokens.png")


# ---------------------------------------------------------------- bivariate --
def fig_length_by_emotion(train: pd.DataFrame) -> str:
    data = [train.loc[train["label_name"] == c, "n_words"] for c in LABEL_NAMES]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    ax.set_xticks(range(1, len(LABEL_NAMES) + 1))
    ax.set_xticklabels(LABEL_NAMES)
    for patch, name in zip(bp["boxes"], LABEL_NAMES):
        patch.set_facecolor(PALETTE[name])
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
    ax.set_title("Message length by emotion (train)")
    ax.set_ylabel("words")
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "fig04_length_by_emotion.png")


def fig_negation_by_emotion(train: pd.DataFrame) -> str:
    rate = (train.groupby("label_name")["has_negation"].mean() * 100).reindex(LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(rate.index, rate.values, color=[PALETTE[c] for c in rate.index])
    for bar, value in zip(bars, rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Share of messages containing a negation, by emotion")
    ax.set_ylabel("% of messages")
    ax.set_ylim(0, rate.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "fig05_negation_by_emotion.png")


def distinctive_tokens(train: pd.DataFrame, top_n: int = 12) -> dict[str, list]:
    """Log-odds of each token for a class vs the rest of the corpus.

    Raw frequency would just return 'i', 'feel', 'the' for every class. Log-odds
    with add-one smoothing surfaces what is *characteristic* of a class.
    """
    per_class = {
        name: Counter(" ".join(train.loc[train["label_name"] == name, "text_clean"]).split())
        for name in LABEL_NAMES
    }
    overall = Counter()
    for counter in per_class.values():
        overall.update(counter)

    total_all = sum(overall.values())
    result = {}
    for name, counter in per_class.items():
        total_class = sum(counter.values()) or 1
        scored = []
        for token, count in counter.items():
            if count < 5:
                continue
            p_class = count / total_class
            p_rest = (overall[token] - count + 1) / (total_all - total_class + 1)
            scored.append((token, math.log(p_class / p_rest)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        result[name] = scored[:top_n]
    return result


def fig_distinctive_tokens(train: pd.DataFrame) -> str:
    tokens = distinctive_tokens(train)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, name in zip(axes.ravel(), LABEL_NAMES):
        pairs = tokens[name][::-1]
        ax.barh([t for t, _ in pairs], [s for _, s in pairs], color=PALETTE[name])
        ax.set_title(name)
        ax.set_xlabel("log-odds vs rest of corpus")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Most distinctive tokens per emotion (train)", fontsize=13)
    return save(fig, "fig06_distinctive_tokens.png")


# ------------------------------------------------------------------- target --
def fig_split_consistency(frames: dict[str, pd.DataFrame]) -> str:
    shares = pd.DataFrame(
        {
            split: (df["label_name"].value_counts(normalize=True) * 100).reindex(LABEL_NAMES)
            for split, df in frames.items()
        }
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.26
    positions = range(len(LABEL_NAMES))
    for i, split in enumerate(SPLITS):
        ax.bar([p + i * width for p in positions], shares[split].values,
               width=width, label=split)
    ax.set_xticks([p + width for p in positions])
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_ylabel("% of split")
    ax.set_title("Class share per split - are train / validation / test comparable?")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return save(fig, "fig07_split_consistency.png"), shares


def fig_class_similarity(train: pd.DataFrame) -> str:
    """Cosine similarity between the token profiles of each class.

    High similarity predicts which pairs the classifiers will confuse, before a
    single model is trained.
    """
    profiles = {
        name: Counter(" ".join(train.loc[train["label_name"] == name, "text_clean"]).split())
        for name in LABEL_NAMES
    }
    vocab = sorted({t for c in profiles.values() for t in c})
    index = {t: i for i, t in enumerate(vocab)}

    vectors = {}
    for name, counter in profiles.items():
        vec = [0.0] * len(vocab)
        total = sum(counter.values()) or 1
        for token, count in counter.items():
            vec[index[token]] = count / total
        vectors[name] = vec

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    matrix = pd.DataFrame(
        [[cosine(vectors[r], vectors[c]) for c in LABEL_NAMES] for r in LABEL_NAMES],
        index=LABEL_NAMES, columns=LABEL_NAMES,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    for i in range(len(LABEL_NAMES)):
        for j in range(len(LABEL_NAMES)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix.iloc[i, j] < 0.75 else "black", fontsize=8)
    ax.set_title("Vocabulary similarity between emotions")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return save(fig, "fig08_class_similarity.png"), matrix


# --------------------------------------------------------------------- main --
def main() -> int:
    ensure_dirs()
    frames = load()
    train = frames["train"]

    print("generating figures")
    fig_class_distribution(train)
    fig_length_distribution(train)
    fig_top_tokens(train)
    fig_length_by_emotion(train)
    fig_negation_by_emotion(train)
    fig_distinctive_tokens(train)
    _, shares = fig_split_consistency(frames)
    _, similarity = fig_class_similarity(train)

    # ---- written report -----------------------------------------------------
    counts = train["label_name"].value_counts().reindex(LABEL_NAMES)
    imbalance = counts.max() / counts.min()
    max_drift = (shares.max(axis=1) - shares.min(axis=1)).max()

    off_diagonal = [
        (r, c, similarity.loc[r, c])
        for i, r in enumerate(LABEL_NAMES)
        for c in LABEL_NAMES[i + 1:]
    ]
    off_diagonal.sort(key=lambda triple: triple[2], reverse=True)

    lines = [
        "# Exploratory Data Analysis",
        "",
        "Generated by `python -m src.data.eda`. Every figure is reproducible.",
        "",
        "## 1. Dataset size",
        "",
        "| split | messages |",
        "|---|---|",
    ]
    lines += [f"| {s} | {len(df):,} |" for s, df in frames.items()]

    lines += [
        "",
        "## 2. Class distribution (univariate)",
        "",
        "![class distribution](figures/fig01_class_distribution.png)",
        "",
        "| emotion | train messages | share |",
        "|---|---|---|",
    ]
    lines += [
        f"| {name} | {counts[name]:,} | {100 * counts[name] / counts.sum():.1f}% |"
        for name in counts.sort_values(ascending=False).index
    ]
    lines += [
        "",
        f"The largest class is **{counts.idxmax()}** and the smallest is "
        f"**{counts.idxmin()}**, a ratio of **{imbalance:.1f}:1**.",
        "",
        "This is the central fact of the project. A model that never predicts "
        f"`{counts.idxmin()}` still scores "
        f"{100 * (1 - counts.min() / counts.sum()):.1f}% accuracy, which is why "
        "accuracy is not used as the headline metric anywhere in this work. "
        "Macro F1 and per-class recall are reported instead.",
        "",
        "## 3. Message length (univariate)",
        "",
        "![length distribution](figures/fig02_length_distribution.png)",
        "",
        f"Median {train['n_words'].median():.0f} words "
        f"({train['n_chars'].median():.0f} characters); "
        f"95th percentile {train['n_words'].quantile(0.95):.0f} words; "
        f"longest {train['n_words'].max():.0f} words.",
        "",
        "Short texts have a practical consequence: TF-IDF vectors will be very "
        "sparse, and sequence models need only a modest maximum length, which "
        "keeps training cheap.",
        "",
        "## 4. Vocabulary (univariate)",
        "",
        "![top tokens](figures/fig03_top_tokens.png)",
        "",
        "## 5. Length and negation by emotion (bivariate)",
        "",
        "![length by emotion](figures/fig04_length_by_emotion.png)",
        "",
        "![negation by emotion](figures/fig05_negation_by_emotion.png)",
        "",
        f"Overall {100 * train['has_negation'].mean():.1f}% of messages contain a "
        "negation token. Negation reverses sentiment while leaving the "
        "surrounding vocabulary intact, so it is a standard source of "
        "false positives for bag-of-words models and a useful slice for error "
        "analysis.",
        "",
        "## 6. Distinctive vocabulary per emotion (bivariate)",
        "",
        "![distinctive tokens](figures/fig06_distinctive_tokens.png)",
        "",
        "Scored by log-odds against the rest of the corpus rather than raw "
        "frequency, which would return the same function words for every class.",
        "",
        "## 7. Split consistency (target-focused)",
        "",
        "![split consistency](figures/fig07_split_consistency.png)",
        "",
        f"Largest difference in class share between any two splits: "
        f"**{max_drift:.1f} percentage points**. The splits are comparable, so "
        "validation results should transfer to test.",
        "",
        "## 8. Class separability (target-focused)",
        "",
        "![class similarity](figures/fig08_class_similarity.png)",
        "",
        "Cosine similarity between the token-frequency profile of each emotion. "
        "The most overlapping pairs are:",
        "",
        "| pair | similarity |",
        "|---|---|",
    ]
    lines += [f"| {a} / {b} | {score:.3f} |" for a, b, score in off_diagonal[:5]]
    lines += [
        "",
        f"**{off_diagonal[0][0]}** and **{off_diagonal[0][1]}** share the most "
        "vocabulary, so they are the pair most likely to be confused. This is a "
        "prediction made before any model was trained; the confusion matrices in "
        "the modelling sections should be checked against it.",
        "",
        "## 9. Implications for modelling",
        "",
        "1. Report macro F1 and per-class recall, never accuracy alone.",
        "2. Apply class weighting or resampling and justify the choice.",
        f"3. Watch the {off_diagonal[0][0]}/{off_diagonal[0][1]} boundary "
        "specifically in error analysis.",
        "4. Keep negation in the text; removing stopwords would delete it.",
        "5. Short messages mean a small maximum sequence length is sufficient.",
        "",
    ]

    path = REPORTS_DIR / "eda_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote reports/{path.name}")
    print(f"imbalance ratio {imbalance:.1f}:1   max split drift {max_drift:.1f}pp")
    print(f"most similar pair: {off_diagonal[0][0]}/{off_diagonal[0][1]} "
          f"({off_diagonal[0][2]:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
