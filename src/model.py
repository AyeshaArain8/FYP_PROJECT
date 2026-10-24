from tensorflow import keras
from tensorflow.keras import layers


def build_model(num_classes: int, seq_len: int, feature_dim: int,
                 lstm_units: int = 128, dropout: float = 0.4, learning_rate: float = 1e-3):
    """
    BiLSTM sequence classifier for isolated PSL sign recognition.

    Input:  (batch, seq_len, feature_dim) landmark sequences
    Output: (batch, num_classes) softmax probabilities

    Masking(mask_value=0.0) lets the model ignore padded frames (the ones we
    added in preprocessing when a sample was shorter than seq_len).
    """
    inputs = keras.Input(shape=(seq_len, feature_dim))
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units // 2))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
