"""Model smoke tests.

Not accuracy tests - those belong in reports/classical_ml/REPORT.md and the
deep learning evaluation. These check the basic contract every model must
satisfy to be safely deployed: it loads, it returns one of the six valid
labels, and it doesn't silently return the same label for everything.

Run from the project root:

    pytest tests/test_model_smoke.py -v

Requires model files to be present in models/ - these tests are skipped
(not failed) if a given model isn't found, since not every contributor's
machine will have every model file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.schema import LABEL_NAMES

MODELS_DIR = Path("models")

CLASSICAL_MODEL_PATH = MODELS_DIR / "calibrated_logreg_unigrams.joblib"
CLASSICAL_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer_unigrams.joblib"

DEEP_MODEL_PATH = MODELS_DIR / "embedding_model.h5"
VOCAB_PATH = MODELS_DIR / "vocabulary.pkl"

SAMPLE_TEXTS = [
    "i am so happy today everything is wonderful",
    "i am furious about this and cannot believe it happened",
    "i feel so scared and anxious about tomorrow",
]


def _classical_available() -> bool:
    return CLASSICAL_MODEL_PATH.exists() and CLASSICAL_VECTORIZER_PATH.exists()


def _deep_available() -> bool:
    return DEEP_MODEL_PATH.exists() and VOCAB_PATH.exists()


@pytest.mark.skipif(not _classical_available(), reason="classical model files not present")
class TestClassicalModelContract:
    @pytest.fixture(scope="class")
    def model_and_vectorizer(self):
        import joblib

        model = joblib.load(CLASSICAL_MODEL_PATH)
        vectorizer = joblib.load(CLASSICAL_VECTORIZER_PATH)
        return model, vectorizer

    def test_predicts_valid_label_for_every_sample(self, model_and_vectorizer):
        model, vectorizer = model_and_vectorizer
        for text in SAMPLE_TEXTS:
            X = vectorizer.transform([text])
            pred = model.predict(X)[0]
            assert 0 <= pred < len(LABEL_NAMES), (
                f"prediction {pred} is outside the valid label range "
                f"0-{len(LABEL_NAMES) - 1} for input: {text!r}"
            )

    def test_does_not_return_the_same_label_for_every_input(self, model_and_vectorizer):
        """A model that always predicts the majority class technically
        satisfies the label-range test above but is not doing its job.
        This is a weak but cheap sanity check against that failure mode.
        """
        model, vectorizer = model_and_vectorizer
        preds = set()
        for text in SAMPLE_TEXTS:
            X = vectorizer.transform([text])
            preds.add(model.predict(X)[0])
        assert len(preds) > 1, (
            "model predicted the same label for every sample text - "
            "check the model and vectorizer were saved from the same run"
        )


@pytest.mark.skipif(not _deep_available(), reason="deep learning model files not present")
class TestDeepModelContract:
    @pytest.fixture(scope="class")
    def model(self):
        import pickle

        import tensorflow as tf

        model = tf.keras.models.load_model(str(DEEP_MODEL_PATH))

        with open(VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)
        for layer in model.layers:
            if hasattr(layer, "set_vocabulary"):
                layer.set_vocabulary(vocab)

        return model

    def test_vocabulary_is_not_empty(self, model):
        """Regression test for the .h5 vocabulary-loss bug: a model whose
        TextVectorization layer has no vocabulary will raise
        'Table not initialized' at predict time rather than failing here,
        which makes it worth checking explicitly and early.
        """
        for layer in model.layers:
            if hasattr(layer, "get_vocabulary"):
                vocab = layer.get_vocabulary()
                assert len(vocab) > 1, (
                    "TextVectorization layer has an empty or near-empty "
                    "vocabulary - predictions will raise 'Table not "
                    "initialized'. Check that vocabulary.pkl matches this "
                    "model and was applied before this test ran."
                )

    def test_predicts_valid_label_for_every_sample(self, model):
        import numpy as np
        import tensorflow as tf

        for text in SAMPLE_TEXTS:
            input_tensor = tf.constant([text], dtype=tf.string)
            probs = model.predict(input_tensor, verbose=0)[0]
            assert len(probs) == len(LABEL_NAMES), (
                f"model output has {len(probs)} classes, expected "
                f"{len(LABEL_NAMES)}"
            )
            pred = int(np.argmax(probs))
            assert 0 <= pred < len(LABEL_NAMES)
