"""
Export Trained Models from Notebook
Run this at the end of 01_Deep_Learning_Baseline.ipynb
"""
import os
import json
import pickle
from pathlib import Path


def export_models(model, lstm_model, vectorizer, output_dir="../../models"):
    """
    Export trained models to disk for deployment.

    Args:
        model: Trained embedding-based model
        lstm_model: Trained LSTM model
        vectorizer: Fitted TextVectorization layer
        output_dir: Directory to save models
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Save Embedding-based model
    embedding_path = output_path / "embedding_model.h5"
    model.save(str(embedding_path))
    print(f"✓ Saved Embedding model: {embedding_path}")

    # 2. Save LSTM model
    lstm_path = output_path / "lstm_model.h5"
    lstm_model.save(str(lstm_path))
    print(f"✓ Saved LSTM model: {lstm_path}")

    # 3. Save Vectorizer config (for reference)
    vectorizer_config = {
        'max_tokens': vectorizer.max_tokens,
        'sequence_length': vectorizer.output_sequence_length,
        'output_mode': vectorizer.output_mode
    }
    config_path = output_path / "vectorizer_config.json"
    with open(config_path, 'w') as f:
        json.dump(vectorizer_config, f, indent=2)
    print(f"✓ Saved Vectorizer config: {config_path}")

    # 4. Save vocabulary (critical for reproducibility)
    vocab_path = output_path / "vocabulary.pkl"
    with open(vocab_path, 'wb') as f:
        pickle.dump(vectorizer.get_vocabulary(), f)
    print(f"✓ Saved Vocabulary: {vocab_path}")

    print(f"\n✓ All models exported to {output_path.absolute()}")
    print("\nFiles created:")
    for file in output_path.glob("*"):
        print(f"  - {file.name}")


if __name__ == "__main__":
    # This would be called at the end of the notebook:
    # export_models(model, lstm_model, vectorizer)
    pass