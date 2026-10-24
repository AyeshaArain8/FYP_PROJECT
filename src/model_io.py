"""
Robust model loading helper.

keras.models.load_model() on an .h5 file can fail with errors like
"Unknown layer: 'NotEqual'" if the Keras/TensorFlow version differs even
slightly between saving and loading -- this is a known Keras 3 compatibility
issue with the legacy H5 format, especially for models using a Masking
layer (which is exactly what our BiLSTM model uses).

The fix: rebuild the architecture fresh from model.py (cheap, deterministic,
version-independent) and load ONLY the weights from the .h5 file, instead of
asking Keras to reconstruct the whole computation graph from stored config.
This sidesteps the bug entirely and does not require retraining.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from model import build_model


def load_trained_model():
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    num_classes = len(label_map["idx_to_class"])

    model = build_model(
        num_classes, config.SEQ_LEN, config.FEATURE_DIM,
        learning_rate=config.LEARNING_RATE,
    )
    model.load_weights(str(config.MODEL_PATH))
    return model
