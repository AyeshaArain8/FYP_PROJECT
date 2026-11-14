"""
Video sign recognition: splits a clip at short gaps where no hand is detected,
then classifies each isolated sign segment. The underlying model is still a
per-sign classifier; it does not understand sentence structure.

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
from sequence_utils import prepare_live_sequence
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


def split_detected_segments(feats: np.ndarray, max_gap: int = 8,
                            min_active_frames: int = 3) -> list:
    """Split a clip where hand landmarks disappear between isolated signs."""
    if len(feats) == 0:
        return []

    active = np.any(np.abs(feats) > 1e-8, axis=1)
    starts = np.where(active & ~np.r_[False, active[:-1]])[0]
    ends = np.where(active & ~np.r_[active[1:], False])[0]
    intervals = []
    for start, end in zip(starts, ends):
        start, end = int(start), int(end)
        if intervals and start - intervals[-1][1] - 1 <= max_gap:
            intervals[-1] = (intervals[-1][0], end)
        else:
            intervals.append((start, end))

    segments = []
    for start, end in intervals:
        segment = feats[start:end + 1]
        active_count = int(np.any(np.abs(segment) > 1e-8, axis=1).sum())
        if active_count >= min_active_frames:
            segments.append(segment)
    return segments


def sliding_window_predict(model, classes, feats: np.ndarray,
                            window: int = config.SEQ_LEN, stride: int = 10,
                            conf_threshold: float = 0.65) -> list:
    """Classify one isolated sign per detected-hand segment in a video."""
    del stride  # retained for call compatibility; segments replace overlapping windows
    predictions = []
    for segment in split_detected_segments(feats):
        prepared = prepare_live_sequence(segment, window)
        if prepared is None:
            continue

        probs = model.predict(np.expand_dims(prepared, 0), verbose=0)[0]
        idx = int(probs.argmax())
        if float(probs[idx]) >= conf_threshold:
            predictions.append(classes[idx])
    return predictions


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
