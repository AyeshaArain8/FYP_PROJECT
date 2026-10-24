"""
Continuous LIVE webcam demo -- no key press needed. Keep signing in front of
the camera and it automatically detects each sign, builds up a sentence on
screen, and speaks each new word out loud.

USAGE
    python src/predict_live_continuous.py

Controls:
    c  -> clear the current sentence
    q  -> quit
"""

import json
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from landmark_utils import HandLandmarkExtractor
from gloss_to_english import gloss_to_english
from tts import speak
from model_io import load_trained_model

# How often (in frames) to run a classification on the current buffer.
PREDICT_EVERY_N_FRAMES = 8

# Only accept a prediction if the model is at least this confident.
CONFIDENCE_THRESHOLD = 0.90

# The SAME word must be predicted this many times in a row (consecutive
# checks, each CONFIDENCE_THRESHOLD or higher) before it's accepted into
# the sentence. This is what makes the system "wait until sure" instead of
# reacting to a single lucky high-confidence frame -- much more reliable.
STABILITY_REQUIRED = 4

# Don't add the same word again within this many seconds (avoids spamming
# the sentence with the same sign repeated while you hold your hand still).
REPEAT_COOLDOWN_SECONDS = 3.0


def speak_async(text: str):
    """Speak without blocking the video loop."""
    threading.Thread(target=speak, args=(text,), daemon=True).start()


def main():
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    classes = label_map["idx_to_class"]

    print("Loading model...")
    model = load_trained_model()
    extractor = HandLandmarkExtractor()

    buffer = deque(maxlen=config.SEQ_LEN)
    sentence_glosses = []
    last_confirmed_word = None
    last_confirmed_time = 0.0
    frame_count = 0
    last_prediction_text = ""

    # Tracks how many consecutive checks the current candidate word has
    # been predicted with high confidence.
    candidate_word = None
    candidate_streak = 0

    cap = cv2.VideoCapture(0)
    print("Live mode running. Start signing. Press 'c' to clear, 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        feat = extractor.extract(frame)
        buffer.append(feat)
        frame_count += 1

        if len(buffer) == config.SEQ_LEN and frame_count % PREDICT_EVERY_N_FRAMES == 0:
            seq = np.expand_dims(np.array(buffer), axis=0)
            probs = model.predict(seq, verbose=0)[0]
            idx = int(probs.argmax())
            confidence = float(probs[idx])
            word = classes[idx]
            last_prediction_text = f"{word} ({confidence:.2f})  streak: {candidate_streak}/{STABILITY_REQUIRED}"

            if confidence >= CONFIDENCE_THRESHOLD:
                if word == candidate_word:
                    candidate_streak += 1
                else:
                    candidate_word = word
                    candidate_streak = 1
            else:
                candidate_word = None
                candidate_streak = 0

            if candidate_streak >= STABILITY_REQUIRED:
                now = time.time()
                is_new = (candidate_word != last_confirmed_word) or (
                    now - last_confirmed_time > REPEAT_COOLDOWN_SECONDS
                )
                if is_new:
                    sentence_glosses.append(candidate_word)
                    last_confirmed_word = candidate_word
                    last_confirmed_time = now
                    speak_async(candidate_word)
                candidate_word = None
                candidate_streak = 0

        # ---- on-screen overlay ----
        english = gloss_to_english(sentence_glosses)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Last guess: {last_prediction_text}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {english}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("PSL Live Continuous Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sentence_glosses.clear()
            last_confirmed_word = None
            candidate_word = None
            candidate_streak = 0
            print("Sentence cleared.")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    print("\nFinal sentence:", gloss_to_english(sentence_glosses))


if __name__ == "__main__":
    main()
