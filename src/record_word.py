"""Capture camera examples for an existing or new PSL sign.

Press R once per example. After a short countdown, the script records one
fixed-length sign automatically and saves it. Press Q to finish early.
"""

import argparse
import ctypes
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from landmark_utils import HandLandmarkExtractor
from sequence_utils import trim_dead_frames

COUNTDOWN_SECONDS = 2.0
RECORD_SECONDS = 1.8
MIN_ACTIVE_FRAMES = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word", required=True, help="label, e.g. good_morning")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    word = re.sub(r"\s+", "_", args.word.strip().lower())
    if not re.fullmatch(r"[a-z0-9]+(?:[_-][a-z0-9]+)*", word):
        parser.error("--word may contain letters, numbers, '_' or '-' only")
    if args.samples < 1:
        parser.error("--samples must be positive")

    output_dir = config.DATA_RAW_DIR / "custom_samples" / word
    output_dir.mkdir(parents=True, exist_ok=True)
    old_ids = [int(path.stem.split("_")[-1]) for path in output_dir.glob("sample_*.npy")
               if path.stem.split("_")[-1].isdigit()]
    next_id = max(old_ids, default=0) + 1

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"Could not open camera {args.camera}")

    window = "Record a PSL sign"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        display_size = (ctypes.windll.user32.GetSystemMetrics(0),
                        ctypes.windll.user32.GetSystemMetrics(1))
    except (AttributeError, OSError):
        display_size = (1280, 720)

    state = "ready"
    deadline = 0.0
    current = []
    saved = 0
    message = "Press R once; wait for countdown, then perform the sign"
    print("Press R once for each example. It counts down, records, and saves automatically. Q finishes.")
    try:
        while saved < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            feature, results = extractor.extract_with_raw(frame)
            detected = extractor.draw_landmarks(frame, results)

            if state == "recording":
                current.append(feature)
                if now >= deadline:
                    sample = trim_dead_frames(np.asarray(current, dtype=np.float32))
                    active_count = int(np.any(np.abs(sample) > 1e-8, axis=1).sum())
                    if active_count >= MIN_ACTIVE_FRAMES:
                        path = output_dir / f"sample_{next_id:04d}.npy"
                        np.save(path, sample)
                        next_id += 1
                        saved += 1
                        message = f"Saved example {saved}/{args.samples}"
                    else:
                        message = "Too few detected frames; press R and try again"
                    current = []
                    state = "ready"
            elif state == "countdown" and now >= deadline:
                state = "recording"
                deadline = now + RECORD_SECONDS
                current = []
                current.append(feature)

            hand_status = "HAND DETECTED" if detected else "NO HAND DETECTED"
            status_color = (0, 255, 0) if detected else (0, 0, 255)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 112), (18, 22, 30), -1)
            cv2.putText(frame, f"Word: {word}     {saved}/{args.samples} examples",
                        (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if state == "countdown":
                instruction = f"Get ready: {max(1, int(deadline - now + 0.99))}"
            elif state == "recording":
                instruction = "RECORDING NOW — perform the full sign"
            else:
                instruction = message
            cv2.putText(frame, instruction, (18, 76), cv2.FONT_HERSHEY_SIMPLEX,
                        0.72, (0, 235, 255) if state != "recording" else (0, 0, 255), 2)
            cv2.putText(frame, hand_status, (18, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2)
            cv2.imshow(window, cv2.resize(frame, display_size, interpolation=cv2.INTER_LINEAR))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("r"), ord("R")) and state == "ready":
                state = "countdown"
                deadline = time.monotonic() + COUNTDOWN_SECONDS
                message = "Get ready to perform one complete sign"
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
    print(f"Recorded {saved} usable examples for '{word}'.")


if __name__ == "__main__":
    main()
