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
import ctypes
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from landmark_utils import HandLandmarkExtractor
from sequence_utils import prepare_live_sequence
from gloss_to_english import gloss_to_english
from tts import speak
from model_io import load_trained_model

PREDICT_EVERY_N_FRAMES = 8
CONFIDENCE_THRESHOLD = 0.85
STABILITY_REQUIRED = 2
REPEAT_COOLDOWN_SECONDS = 2.5
NO_HAND_RESET_FRAMES = 6


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
    frame_count = 0
    sentence_glosses = []
    last_prediction_text = "Show a sign"
    candidate_word = None
    candidate_streak = 0
    last_confirmed_word = None
    last_confirmed_time = 0.0
    no_hand_streak = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Request a useful camera resolution and make the preview large enough to
    # check hand placement and landmark tracking while signing.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    window_name = "PSL Live Continuous Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setWindowProperty(
        window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )
    try:
        display_size = (
            ctypes.windll.user32.GetSystemMetrics(0),
            ctypes.windll.user32.GetSystemMetrics(1),
        )
    except (AttributeError, OSError):
        display_size = (1280, 720)
    print("Live recognition is automatic. Press 'c' to clear and 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        feat, hand_results = extractor.extract_with_raw(frame)
        hand_detected = extractor.draw_landmarks(frame, hand_results)
        buffer.append(feat)
        frame_count += 1
        if hand_detected:
            no_hand_streak = 0
        else:
            no_hand_streak += 1
            if no_hand_streak >= NO_HAND_RESET_FRAMES:
                candidate_word = None
                candidate_streak = 0
                last_prediction_text = "No hand detected"

        if len(buffer) == config.SEQ_LEN and frame_count % PREDICT_EVERY_N_FRAMES == 0:
            # Never classify an old buffered gesture while the current frame
            # has no detected hand. Keep the rolling history so short tracker
            # drops do not force the signer to restart the gesture.
            if hand_detected:
                prepared = prepare_live_sequence(np.asarray(buffer), config.SEQ_LEN)
            else:
                prepared = None
            if prepared is not None:
                probabilities = model.predict(prepared[np.newaxis, ...], verbose=0)[0]
                index = int(probabilities.argmax())
                confidence = float(probabilities[index])
                word = classes[index]
                last_prediction_text = f"{word} ({confidence:.2f})"
                threshold = 0.995 if word == "nothing" else CONFIDENCE_THRESHOLD
                needed = 3 if word == "nothing" else STABILITY_REQUIRED

                if confidence >= threshold:
                    if word == candidate_word:
                        candidate_streak += 1
                    else:
                        candidate_word = word
                        candidate_streak = 1
                else:
                    candidate_word = None
                    candidate_streak = 0

                if candidate_streak >= needed:
                    now = time.time()
                    if word != last_confirmed_word or now - last_confirmed_time >= REPEAT_COOLDOWN_SECONDS:
                        sentence_glosses.append(word)
                        speak_async(word)
                        last_confirmed_word = word
                        last_confirmed_time = now
                        last_prediction_text = f"Output: {word} ({confidence:.2f})"
                    candidate_word = None
                    candidate_streak = 0

        # ---- on-screen overlay ----
        english = gloss_to_english(sentence_glosses)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Last guess: {last_prediction_text}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Sentence: {english}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status = "HAND DETECTED" if hand_detected else "NO HAND DETECTED"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        cv2.putText(frame, status, (10, frame.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        display_frame = cv2.resize(frame, display_size, interpolation=cv2.INTER_LINEAR)
        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            sentence_glosses.clear()
            candidate_word = None
            candidate_streak = 0
            last_confirmed_word = None
            print("Sentence cleared.")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()

    print("\nFinal sentence:", gloss_to_english(sentence_glosses))


if __name__ == "__main__":
    main()
