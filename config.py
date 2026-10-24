from pathlib import Path

# ---- Paths ----
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
PLOTS_DIR = OUTPUTS_DIR / "plots"

MODEL_PATH = CHECKPOINT_DIR / "psl_model.h5"
LABEL_MAP_PATH = DATA_PROCESSED_DIR / "label_map.json"
TEMPLATES_PATH = DATA_PROCESSED_DIR / "templates.npz"

# ---- Landmark feature format ----
# 2 hands x 21 MediaPipe landmarks x (x, y, z) = 126.
# IMPORTANT: if you re-extract landmarks yourself (e.g. from a live webcam
# in predict_live.py), the extractor MUST produce vectors in this same
# format or the trained model's weights will not line up with the input.
FEATURE_DIM = 126
SEQ_LEN = 40  # frames per sample after padding/trimming

# ---- Training ----
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3

# Classes with fewer real samples than this are dropped automatically during
# preprocessing (too little data to learn reliably, and drags down reported
# accuracy). Raise this if you want a stricter/cleaner vocabulary, lower it
# if you want to keep more (rarer) words at the cost of some noise.
MIN_SAMPLES_PER_CLASS = 40
