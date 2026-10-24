"""
DEBUG version of the live demo -- shows the TOP 5 predictions with their
confidence scores every cycle, instead of just the single best guess. This
is for figuring out WHY live predictions differ from the high test-set
accuracy (e.g. is the correct word close-but-not-quite, or is it not even
in the top 5 -- which would point to a feature/hand-ordering bug instead
of just an imperfect gesture).

USAGE
    python src/predict_live_debug.py
    (press 'q' to quit)
"""

import json
import sys
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from landmark_utils import HandLandmarkExtractor
from model_io import load_trained_model

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def main():
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    classes = label_map["idx_to_class"]

    print("Loading model...")
    model = load_trained_model()
    extractor = HandLandmarkExtractor()

    buffer = deque(maxlen=config.SEQ_LEN)
    cap = cv2.VideoCapture(0)
    frame_count = 0
    top5_text = []

    print("Debug mode running. Watch the terminal + on-screen overlay. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        feat, raw_results = extractor.extract_with_raw(frame)
        buffer.append(feat)
        frame_count += 1

        # Draw whatever MediaPipe actually detected, so you can SEE it directly
        # instead of guessing from the L/R text status.
        if raw_results.multi_hand_landmarks:
            for hand_landmarks in raw_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Report which hand(s) are currently detected -- useful for spotting
        # a left/right ordering mismatch.
        left_active = np.any(np.abs(feat[:63]) > 1e-6)
        right_active = np.any(np.abs(feat[63:]) > 1e-6)
        hand_status = f"L:{'Y' if left_active else '-'}  R:{'Y' if right_active else '-'}"

        if len(buffer) == config.SEQ_LEN and frame_count % 8 == 0:
            seq = np.expand_dims(np.array(buffer), axis=0)
            probs = model.predict(seq, verbose=0)[0]
            top5_idx = probs.argsort()[-5:][::-1]
            top5_text = [f"{classes[i]}: {probs[i]:.2f}" for i in top5_idx]
            print(hand_status, "|", ", ".join(top5_text))

        cv2.rectangle(frame, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
        cv2.putText(frame, f"Hands detected -> {hand_status}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        for i, line in enumerate(top5_text):
            cv2.putText(frame, line, (10, 55 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("PSL Debug - Top 5 predictions", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()
