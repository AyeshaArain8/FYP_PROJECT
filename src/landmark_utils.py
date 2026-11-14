"""Extract dataset-compatible 126-value vectors from live camera frames.

The processed dataset puts almost every one-hand sample in slot two. When
both hands are present, the first slot usually contains the hand farther right
in image coordinates. MediaPipe labels are ignored because they can flip when
camera mirroring changes, and duplicate labels can otherwise overwrite a hand.
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
        vector, _ = self.extract_with_raw(frame_bgr)
        return vector

    @staticmethod
    def _pack(results) -> np.ndarray:
        slots = [np.zeros(63, dtype=np.float32), np.zeros(63, dtype=np.float32)]
        detected = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                points = np.asarray(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                detected.append((float(points[:, 0].mean()), points.flatten()))

        if len(detected) == 1:
            slots[1] = detected[0][1]
        elif len(detected) >= 2:
            detected.sort(key=lambda item: item[0], reverse=True)
            slots[0], slots[1] = detected[0][1], detected[1][1]
        return np.concatenate(slots)

    def extract_with_raw(self, frame_bgr: np.ndarray):
        """Return the packed vector and MediaPipe results for drawing."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return self._pack(results), results

    @staticmethod
    def draw_landmarks(frame_bgr: np.ndarray, results) -> bool:
        """Draw green hand landmarks on the camera frame; return hand presence."""
        if not results.multi_hand_landmarks:
            return False
        drawing = mp.solutions.drawing_utils
        landmark_style = drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3)
        connection_style = drawing.DrawingSpec(color=(0, 180, 0), thickness=2)
        for hand_landmarks in results.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=landmark_style,
                connection_drawing_spec=connection_style,
            )
        return True

    def close(self):
        self.hands.close()
