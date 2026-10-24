"""
Extracts a 126-dim landmark vector per frame (2 hands x 21 landmarks x xyz),
matching the format used in the PakistanSignLanguageDatasetV2 dataset.

NOTE ON HAND ORDERING: the original dataset's exact left/right ordering
convention isn't documented in its README. This extractor puts left hand
first (indices 0-62) and right hand second (indices 63-125), which matches
the vast majority of MediaPipe-based PSL/ASL projects. If live predictions
look consistently wrong for asymmetric one-handed signs, try swapping the
order in `extract()` below -- that's the most likely cause.
"""

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands


class HandLandmarkExtractor:
    def __init__(self, max_num_hands=2, min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        left = np.zeros(63, dtype=np.float32)
        right = np.zeros(63, dtype=np.float32)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
                ).flatten()
                label = handedness.classification[0].label  # "Left" or "Right"
                if label == "Left":
                    left = coords
                else:
                    right = coords

        return np.concatenate([left, right])  # (126,)

    def extract_with_raw(self, frame_bgr: np.ndarray):
        """Same as extract(), but also returns MediaPipe's raw results object
        so the caller can draw the detected skeleton for debugging."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        left = np.zeros(63, dtype=np.float32)
        right = np.zeros(63, dtype=np.float32)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                coords = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
                ).flatten()
                label = handedness.classification[0].label
                if label == "Left":
                    left = coords
                else:
                    right = coords

        return np.concatenate([left, right]), results

    def close(self):
        self.hands.close()
