"""
Builds one reference landmark sequence per class, used by text_to_sign.py to
"play back" a sign for the Text -> Sign direction.

USAGE
    python src/build_templates.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def main():
    X_train = np.load(config.DATA_PROCESSED_DIR / "X_train.npy")
    y_train = np.load(config.DATA_PROCESSED_DIR / "y_train.npy")
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    classes = label_map["idx_to_class"]

    templates = {}
    for idx, cls in enumerate(classes):
        idxs = np.where(y_train == idx)[0]
        if len(idxs) == 0:
            continue
        templates[cls] = X_train[idxs[0]]  # first training sample as the reference sign

    np.savez(config.TEMPLATES_PATH, **templates)
    print(f"Saved {len(templates)} sign templates to {config.TEMPLATES_PATH}")


if __name__ == "__main__":
    main()
