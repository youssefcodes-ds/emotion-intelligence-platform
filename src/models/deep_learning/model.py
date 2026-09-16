import tensorflow as tf


def build_lstm_model(vectorizer, vocab_size, embedding_dim=64):
    """
    Build the LSTM emotion classification model.
    """

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(), dtype=tf.string),

        vectorizer,

        tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True
        ),

        tf.keras.layers.LSTM(64),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            6,
            activation="softmax"
        )
    ])

    return model