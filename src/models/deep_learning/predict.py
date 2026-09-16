from pathlib import Path

import tensorflow as tf


# Emotion labels
EMOTION_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}


# Find the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Path to the saved model
MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "deep_learning"
    / "lstm_model.keras"
)


# Load the trained LSTM model
model = tf.keras.models.load_model(MODEL_PATH)


def predict_emotion(text):
    """
    Predict the emotion of a given text.
    """

    prediction = model(
        tf.constant([text])
    ).numpy()

    predicted_class = prediction.argmax(axis=1)[0]

    confidence = prediction[0][predicted_class]

    emotion = EMOTION_LABELS[predicted_class]

    return emotion, float(confidence)