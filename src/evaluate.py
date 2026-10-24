"""
Evaluate the trained model on the held-out test set.

USAGE
    python src/evaluate.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from model_io import load_trained_model


def main():
    X_test = np.load(config.DATA_PROCESSED_DIR / "X_test.npy")
    y_test = np.load(config.DATA_PROCESSED_DIR / "y_test.npy")
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    classes = label_map["idx_to_class"]

    model = load_trained_model()
    y_pred = model.predict(X_test).argmax(axis=1)

    print(f"Test accuracy: {(y_pred == y_test).mean():.4f}\n")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 0.18), max(10, len(classes) * 0.18)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im)
    fig.tight_layout()

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(out_path)
    print(f"\nConfusion matrix saved to {out_path}")


if __name__ == "__main__":
    main()
