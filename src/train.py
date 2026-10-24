"""
Train the baseline isolated-sign classifier.

USAGE
    python src/train.py

Reads the preprocessed data from data/processed/ (already included in this
project -- or re-run src/preprocess.py yourself if you change the filtering
settings in config.py). Saves the best model checkpoint to
outputs/checkpoints/psl_model.h5 and training curves to outputs/plots/.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from model import build_model


def main():
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = np.load(config.DATA_PROCESSED_DIR / "X_train.npy")
    y_train = np.load(config.DATA_PROCESSED_DIR / "y_train.npy")
    X_val = np.load(config.DATA_PROCESSED_DIR / "X_val.npy")
    y_val = np.load(config.DATA_PROCESSED_DIR / "y_val.npy")

    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    num_classes = len(label_map["idx_to_class"])

    print(f"Training on {len(y_train)} samples, validating on {len(y_val)}, "
          f"{num_classes} classes.")

    model = build_model(num_classes, config.SEQ_LEN, config.FEATURE_DIM,
                         learning_rate=config.LEARNING_RATE)
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(config.MODEL_PATH), save_best_only=True,
            monitor="val_accuracy", mode="max"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "training_curves.png")

    print(f"\nBest model saved to {config.MODEL_PATH}")
    print(f"Training curves saved to {config.PLOTS_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()
