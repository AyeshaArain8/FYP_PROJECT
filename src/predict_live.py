"""
Real-time webcam demo: buffers config.SEQ_LEN frames, classifies on demand.

USAGE
    python src/predict_live.py
    (SPACE to classify the current buffer, 'q' to quit)
"""

import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from landmark_utils import HandLandmarkExtractor
from tts import speak
from model_io import load_trained_model


def main():
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    classes = label_map["idx_to_class"]

    model = load_trained_model()
    extractor = HandLandmarkExtractor()

    buffer = deque(maxlen=config.SEQ_LEN)
    cap = cv2.VideoCapture(0)

    print("Press SPACE to classify the buffered gesture, 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        feat = extractor.extract(frame)
        buffer.append(feat)

        cv2.putText(frame, f"buffer: {len(buffer)}/{config.SEQ_LEN}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("PSL live capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and len(buffer) == config.SEQ_LEN:
            seq = np.expand_dims(np.array(buffer), axis=0)  # (1, seq_len, 126)
            probs = model.predict(seq, verbose=0)[0]
            pred_idx = int(probs.argmax())
            confidence = float(probs[pred_idx])
            word = classes[pred_idx]
            print(f"Predicted: {word}  (confidence {confidence:.2f})")
            if confidence > 0.6:
                speak(word)

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()
