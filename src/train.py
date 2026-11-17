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
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from model import build_model


def initialize_expanded_classifier(model, checkpoint_path, num_classes, learning_rate):
    """Copy every old class into a model whose final layer has one new class."""
    if num_classes < 2:
        raise ValueError("Class expansion needs the previous class count plus one new class.")
    old_model = build_model(num_classes - 1, config.SEQ_LEN, config.FEATURE_DIM,
                            learning_rate=learning_rate)
    old_model.load_weights(str(checkpoint_path))
    if len(old_model.layers) != len(model.layers):
        raise ValueError("Old and expanded model architectures do not match.")
    for old_layer, new_layer in zip(old_model.layers[:-1], model.layers[:-1]):
        old_weights = old_layer.get_weights()
        new_weights = new_layer.get_weights()
        if old_weights:
            if [w.shape for w in old_weights] != [w.shape for w in new_weights]:
                raise ValueError(f"Cannot transfer layer '{old_layer.name}' safely.")
            new_layer.set_weights(old_weights)
    old_kernel, old_bias = old_model.layers[-1].get_weights()
    new_kernel, new_bias = model.layers[-1].get_weights()
    if old_kernel.shape[0] != new_kernel.shape[0] or old_kernel.shape[1] != num_classes - 1:
        raise ValueError("The current checkpoint does not match the previous label map.")
    new_kernel[:, :num_classes - 1] = old_kernel
    new_bias[:num_classes - 1] = old_bias
    model.layers[-1].set_weights([new_kernel, new_bias])
    return old_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        action="store_true",
        help="start from the current checkpoint (for adding camera samples to existing classes)",
    )
    parser.add_argument(
        "--expand-classes",
        action="store_true",
        help="transfer the current model body and initialize a larger class-output layer",
    )
    args = parser.parse_args()
    if args.resume and args.expand_classes:
        parser.error("Choose either --resume or --expand-classes, not both.")

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

    learning_rate = config.LEARNING_RATE * (0.1 if args.resume else 1.0)
    if args.expand_classes:
        learning_rate *= 0.5
    model = build_model(num_classes, config.SEQ_LEN, config.FEATURE_DIM,
                         learning_rate=learning_rate)
    if args.resume or args.expand_classes:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Cannot start from prior model; checkpoint not found: {config.MODEL_PATH}")
        if args.expand_classes:
            old_model = initialize_expanded_classifier(
                model, config.MODEL_PATH, num_classes, learning_rate
            )
            print("Transferred all existing classes; initialized only the new output class.")
            del old_model
        else:
            model.load_weights(str(config.MODEL_PATH))
            print(f"Fine-tuning existing checkpoint: {config.MODEL_PATH}")
    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(config.MODEL_PATH), save_best_only=True,
            monitor="val_accuracy", mode="max"
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5 if args.resume else 10,
            restore_best_weights=True, mode="max"
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=min(20, config.EPOCHS) if args.resume else config.EPOCHS,
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
