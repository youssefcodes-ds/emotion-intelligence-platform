from pathlib import Path

import pandas as pd
import tensorflow as tf

from .model import build_lstm_model


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "deep_learning"
    / "lstm_model.keras"
)


# ============================================================
# Configuration
# ============================================================

MAX_TOKENS = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 15


# ============================================================
# Load data
# ============================================================

def load_data():
    """
    Load train, validation and test datasets.
    """

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df["text_clean"]
    y_train = train_df["label"]

    X_val = val_df["text_clean"]
    y_val = val_df["label"]

    X_test = test_df["text_clean"]
    y_test = test_df["label"]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# Build vectorizer
# ============================================================

def build_vectorizer(X_train):
    """
    Create and adapt the TextVectorization layer.
    """

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH
    )

    vectorizer.adapt(X_train)

    return vectorizer


# ============================================================
# Build datasets
# ============================================================

def create_datasets(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
):
    """
    Create TensorFlow datasets for training,
    validation and testing.
    """

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            X_train.to_numpy(),
            y_train.to_numpy()
        )
    ).batch(BATCH_SIZE)

    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            X_val.to_numpy(),
            y_val.to_numpy()
        )
    ).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (
            X_test.to_numpy(),
            y_test.to_numpy()
        )
    ).batch(BATCH_SIZE)

    return train_dataset, val_dataset, test_dataset


# ============================================================
# Train model
# ============================================================

def train_model():
    """
    Train the LSTM emotion classification model
    and save the trained model.
    """

    # Load data
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = load_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Build vectorizer
    vectorizer = build_vectorizer(X_train)

    vocab_size = len(vectorizer.get_vocabulary())

    print(f"Vocabulary size: {vocab_size}")

    # Build datasets
    (
        train_dataset,
        val_dataset,
        test_dataset
    ) = create_datasets(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )

    # Build model
    model = build_lstm_model(
        vectorizer=vectorizer,
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM
    )

    # Compile
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6
    )

    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[
            early_stopping,
            lr_scheduler
        ]
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(
        test_dataset,
        verbose=1
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save model
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save(MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

    return model, history


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    train_model()