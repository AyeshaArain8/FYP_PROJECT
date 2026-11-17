"""Browser-camera PSL recognition for the deployed Streamlit app."""

from collections import deque
from threading import Lock
import time

import av
import cv2
import numpy as np

import config
from landmark_utils import HandLandmarkExtractor
from sequence_utils import prepare_live_sequence


_MODEL_LOCK = Lock()


class LiveSignProcessor:
    """Extract landmarks from WebRTC frames and accumulate stable predictions."""

    def __init__(self, model, classes):
        self.model = model
        self.classes = classes
        self.extractor = HandLandmarkExtractor()
        self.frames = deque(maxlen=config.SEQ_LEN)
        self.frame_count = 0
        self.candidate_word = None
        self.candidate_streak = 0
        self.last_word = None
        self.last_word_time = 0.0
        self.no_hand_streak = 0
        self.lock = Lock()
        self.glosses = []
        self.last_guess = "Show a sign"
        self.hand_detected = False
        self.error = None

    def process_frame(self, video_frame):
        image = video_frame.to_ndarray(format="bgr24")
        try:
            features, results = self.extractor.extract_with_raw(image)
            hand_detected = self.extractor.draw_landmarks(image, results)
            self.frames.append(features)
            self.frame_count += 1

            if hand_detected:
                self.no_hand_streak = 0
            else:
                self.no_hand_streak += 1
                if self.no_hand_streak >= 6:
                    self.candidate_word = None
                    self.candidate_streak = 0
                    self.last_guess = "No hand detected"

            if (
                hand_detected
                and len(self.frames) == config.SEQ_LEN
                and self.frame_count % 8 == 0
            ):
                prepared = prepare_live_sequence(
                    np.asarray(self.frames, dtype=np.float32), config.SEQ_LEN
                )
                if prepared is not None:
                    with _MODEL_LOCK:
                        probabilities = self.model.predict(
                            prepared[np.newaxis, ...], verbose=0
                        )[0]
                    index = int(probabilities.argmax())
                    confidence = float(probabilities[index])
                    word = self.classes[index]
                    self.last_guess = f"{word} ({confidence:.2f})"
                    threshold = 0.995 if word == "nothing" else 0.85
                    needed = 3 if word == "nothing" else 2

                    if confidence >= threshold:
                        if word == self.candidate_word:
                            self.candidate_streak += 1
                        else:
                            self.candidate_word = word
                            self.candidate_streak = 1
                    else:
                        self.candidate_word = None
                        self.candidate_streak = 0

                    if self.candidate_streak >= needed:
                        now = time.time()
                        if (
                            word != self.last_word
                            or now - self.last_word_time >= 2.5
                        ):
                            with self.lock:
                                self.glosses.append(word)
                            self.last_word = word
                            self.last_word_time = now
                            self.last_guess = f"Output: {word} ({confidence:.2f})"
                        self.candidate_word = None
                        self.candidate_streak = 0

            with self.lock:
                self.hand_detected = hand_detected
                last_guess = self.last_guess
                detected = self.hand_detected

            cv2.rectangle(image, (0, 0), (image.shape[1], 72), (20, 35, 48), -1)
            cv2.putText(
                image, f"Guess: {last_guess[:48]}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2,
            )
            status = "HAND DETECTED" if detected else "NO HAND DETECTED"
            color = (0, 255, 0) if detected else (0, 100, 255)
            cv2.putText(
                image, status, (12, image.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
            )
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
            return video_frame

    def snapshot(self):
        with self.lock:
            return list(self.glosses), self.last_guess, self.hand_detected, self.error

    def clear(self):
        with self.lock:
            self.glosses.clear()
            self.error = None
        self.last_word = None
        self.last_word_time = 0.0
        self.candidate_word = None
        self.candidate_streak = 0

