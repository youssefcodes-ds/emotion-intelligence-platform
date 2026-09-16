"""Minimal retrieval-grounded assistant for the emotion platform.

Answers questions like "show me angry messages about billing" by retrieving
the most relevant real messages from the processed dataset - it does not
generate free-text answers or invent statistics. This satisfies the "RAG
knowledge layer" requirement at a basic level: retrieval is real and
grounded, nothing is fabricated.

This is a starting point, not a finished feature. Two extension points are
marked below for whoever picks this up:

  1. Swap the TF-IDF retriever for the BERTopic model already trained in
     notebooks/MawddaNotebooks/01_Deep_Learning_Baseline.ipynb, if the topic
     assignments are wanted instead of plain keyword relevance.
  2. Add an LLM call to phrase the retrieved results as a sentence, IF an
     API key is available - keep the retrieval step as the source of truth
     and never let the LLM add facts the retrieval didn't surface.

Run from the project root:

    python -m rag.assistant "angry messages about billing"
"""

from __future__ import annotations

import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data.schema import LABEL_NAMES
from src.utils.paths import PROCESSED_DIR


class GroundedRetriever:
    """Retrieves real messages relevant to a query. Never generates text."""

    def __init__(self, split: str = "train"):
        path = PROCESSED_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data.build` first."
            )
        self.df = pd.read_csv(path, encoding="utf-8")

        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.df["text_clean"])

    def query(
        self,
        text: str,
        emotion: str | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Return the top_k most relevant real messages for a query.

        Parameters
        ----------
        text : str
            Free-text query, e.g. "billing complaint"
        emotion : str, optional
            Restrict results to one of LABEL_NAMES, e.g. "anger"
        top_k : int
            How many results to return
        """
        pool = self.df
        if emotion is not None:
            if emotion not in LABEL_NAMES:
                raise ValueError(f"emotion must be one of {LABEL_NAMES}, got {emotion!r}")
            mask = pool["label_name"] == emotion
            pool = pool[mask]
            if pool.empty:
                return pool
            sub_matrix = self.matrix[mask.values]
        else:
            sub_matrix = self.matrix

        query_vec = self.vectorizer.transform([text])
        scores = cosine_similarity(query_vec, sub_matrix).ravel()

        top_idx = scores.argsort()[::-1][:top_k]
        result = pool.iloc[top_idx].copy()
        result["relevance"] = scores[top_idx]
        return result[["id", "text", "label_name", "relevance"]]

    def summarise(self, text: str, emotion: str | None = None, top_k: int = 5) -> str:
        """Human-readable answer, built only from retrieved rows.

        Deliberately template-based rather than LLM-generated, so nothing
        in the answer can be a fact the retrieval step didn't surface.
        """
        results = self.query(text, emotion=emotion, top_k=top_k)

        if results.empty:
            scope = f" tagged '{emotion}'" if emotion else ""
            return f"No messages{scope} matched '{text}'."

        lines = [f"Top {len(results)} matches for '{text}'" + (f" ({emotion})" if emotion else "") + ":", ""]
        for _, row in results.iterrows():
            lines.append(f"  [{row['label_name']}, relevance={row['relevance']:.2f}] {row['text'][:100]}")
        return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    query_text = sys.argv[1]
    retriever = GroundedRetriever()
    print(retriever.summarise(query_text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
