
import cv2
import mediapipe as mp
from pathlib import Path

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

video_path = input("Enter video path: ").strip().strip('"')

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Video could not be opened.")
    raise SystemExit

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

output_path = Path("landmarks_check.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(output_path), fourcc, fps, (width, height)
)

total_frames = 0
detected_frames = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, frame = cap.read()

        if not success:
            break

        total_frames += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            detected_frames += 1

            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        status = (
            f"Hands detected: {len(results.multi_hand_landmarks)}"
            if results.multi_hand_landmarks
            else "No hand detected"
        )

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if results.multi_hand_landmarks else (0, 0, 255),
            2
        )

        writer.write(frame)

cap.release()
writer.release()

print("\n--- LANDMARK TEST RESULT ---")
print("Total frames:", total_frames)
print("Frames with detected hands:", detected_frames)
print("Frames without hands:", total_frames - detected_frames)
print("Output video:", output_path.resolve())
