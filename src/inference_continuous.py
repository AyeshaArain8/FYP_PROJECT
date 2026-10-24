"""
Continuous sign recognition: runs a sliding window over a longer video
(multiple signs performed back-to-back) and decodes it into a sequence of
glosses. This is what makes the system handle "sentences" rather than just
one isolated sign at a time -- the underlying model is still a per-sign
classifier (see model.py); the sentence-level behaviour comes from how we
call it here, not from the model itself.

USAGE
    python src/inference_continuous.py path/to/video.mp4
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from landmark_utils import HandLandmarkExtractor
from model_io import load_trained_model


def extract_video_landmarks(video_path: str) -> np.ndarray:
    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(video_path)
    feats = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        feats.append(extractor.extract(frame))
    cap.release()
    extractor.close()
    return np.array(feats)  # (T, 126)


def sliding_window_predict(model, classes, feats: np.ndarray,
                            window: int = config.SEQ_LEN, stride: int = 10,
                            conf_threshold: float = 0.75) -> list:
    if len(feats) == 0:
        return []

    predictions = []
    for start in range(0, max(1, len(feats) - window + 1), stride):
        window_feats = feats[start:start + window]
        if len(window_feats) < window:
            pad = np.repeat(window_feats[-1:], window - len(window_feats), axis=0)
            window_feats = np.concatenate([window_feats, pad], axis=0)

        probs = model.predict(np.expand_dims(window_feats, 0), verbose=0)[0]
        idx = int(probs.argmax())
        if probs[idx] > conf_threshold:
            predictions.append(classes[idx])

    # De-duplicate consecutive repeats (the same sign gets picked up by
    # multiple overlapping windows while it's being held).
    glosses = []
    for word in predictions:
        if not glosses or glosses[-1] != word:
            glosses.append(word)
    return glosses


def main(video_path: str):
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    classes = label_map["idx_to_class"]
    model = load_trained_model()

    feats = extract_video_landmarks(video_path)
    glosses = sliding_window_predict(model, classes, feats)
    print("Decoded gloss sequence:", glosses)
    return glosses


if __name__ == "__main__":
    main(sys.argv[1])
