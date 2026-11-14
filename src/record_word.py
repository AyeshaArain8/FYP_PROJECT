"""Record live-camera samples for a new PSL vocabulary word.

Run: python src/record_word.py --word please --samples 40
Press R to start/stop each sample and Q to finish.
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from landmark_utils import HandLandmarkExtractor
from sequence_utils import trim_dead_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word", required=True, help="new English word, e.g. good_night")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    word = args.word.strip().lower()
    if not re.fullmatch(r"[a-z0-9]+(?:[_-][a-z0-9]+)*", word):
        parser.error("--word can use lowercase letters, digits, '_' and '-' only")
    if args.samples < 1:
        parser.error("--samples must be positive")

    output_dir = config.DATA_RAW_DIR / "custom_samples" / word
    output_dir.mkdir(parents=True, exist_ok=True)
    old_ids = [int(p.stem.split("_")[-1]) for p in output_dir.glob("sample_*.npy")
               if p.stem.split("_")[-1].isdigit()]
    next_id = max(old_ids, default=0) + 1

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        extractor.close()
        raise RuntimeError(f"Could not open camera {args.camera}")

    recording = False
    current = []
    saved = 0
    print("Press R to start/stop one sample. Press Q to finish.")
    try:
        while saved < args.samples:
            ok, frame = cap.read()
            if not ok:
                break
            feature, results = extractor.extract_with_raw(frame)
            detected = extractor.draw_landmarks(frame, results)
            if recording:
                current.append(feature)

            detection = "HAND DETECTED" if detected else "NO HAND DETECTED"
            status = f"{word} | {detection} | saved {saved}/{args.samples} | " + ("RECORDING" if recording else "Press R")
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if recording else ((0, 255, 0) if detected else (0, 0, 255)), 2)
            cv2.imshow("Record a PSL word", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                if not recording:
                    current = []
                    recording = True
                else:
                    recording = False
                    sample = trim_dead_frames(np.asarray(current, dtype=np.float32))
                    if len(sample) < 3:
                        print("Skipped: need at least 3 frames with a detected hand.")
                    else:
                        path = output_dir / f"sample_{next_id:04d}.npy"
                        np.save(path, sample)
                        next_id += 1
                        saved += 1
                        print(f"Saved {saved}/{args.samples}: {path}")
                    current = []
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
    print(f"Recorded {saved} usable samples for '{word}'.")


if __name__ == "__main__":
    main()
