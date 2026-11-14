
"""
Robust model loading helper.

Rebuilds the model architecture from model.py and loads only
the trained weights from the H5 file.

Avoids Keras H5 deserialization errors such as:
"Unknown layer: 'NotEqual'"

No retraining is required.
"""

import json
import sys
from pathlib import Path

# Project directories
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Add both directories to Python path
for directory in (str(PROJECT_ROOT), str(SRC_DIR)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

import config
from model import build_model


def load_trained_model():
    """
    Rebuild model architecture and load existing trained weights.
    """

    with open(
        config.LABEL_MAP_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        label_map = json.load(f)

    num_classes = len(label_map["idx_to_class"])

    model = build_model(
        num_classes,
        config.SEQ_LEN,
        config.FEATURE_DIM,
        learning_rate=config.LEARNING_RATE,
    )

    model.load_weights(str(config.MODEL_PATH))

    return model
