"""
Preprocess "PakistanSignLanguageDatasetV2" (archive__4_.zip) into fixed-length
landmark sequences ready for training.

Place archive__4_.zip in data/raw/ before running this (see README.md).

WHAT THIS SCRIPT DOES
1. Reads landmark frames directly out of the zip (no need to extract 400k+
   tiny files onto disk).
2. Groups frames into samples: (source_device, class_name, sample_id) -> a
   sequence of frames, sorted in the correct order.
3. Drops junk/test classes (names containing "test") and any class with
   fewer real samples than config.MIN_SAMPLES_PER_CLASS -- these either
   aren't real vocabulary or don't have enough data to learn reliably.
4. Trims "dead" frames at the start/end of each sample where MediaPipe
   detected no hand at all.
5. Pads or truncates every sample to config.SEQ_LEN frames.
6. Splits into train/val/test, stratified by class.
7. Saves everything to data/processed/.

USAGE
    python src/preprocess.py --zip data/raw/archive__4_.zip
"""

import argparse
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

DEFAULT_SOURCES = ["laptop_data", "mobile_data", "webcam_data"]

PATH_RE = re.compile(
    r"PakistanSignLanguageDatasetV2/PakistanSignLanguageDatasetV2/"
    r"(?P<source>\w+)/(?P<cls>[^/]+)/(?P<sample>\d+)/(?P<frame>\d+)\.npy$"
)


def discover_samples(zf: zipfile.ZipFile, sources):
    samples = defaultdict(dict)
    for name in zf.namelist():
        m = PATH_RE.search(name)
        if not m:
            continue
        source = m.group("source")
        if source not in sources:
            continue
        cls = m.group("cls")
        sample_id = int(m.group("sample"))
        frame_idx = int(m.group("frame"))
        samples[(source, cls, sample_id)][frame_idx] = name
    return samples


def load_sequence(zf: zipfile.ZipFile, frame_map: dict) -> np.ndarray:
    frame_indices = sorted(frame_map.keys())
    frames = [np.load(zf.open(frame_map[i])) for i in frame_indices]
    return np.stack(frames, axis=0)


def trim_dead_frames(seq: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    is_active = np.any(np.abs(seq) > tolerance, axis=1)
    if not is_active.any():
        return seq
    first = np.argmax(is_active)
    last = len(is_active) - np.argmax(is_active[::-1])
    return seq[first:last]


def pad_or_truncate(seq: np.ndarray, seq_len: int) -> np.ndarray:
    t = seq.shape[0]
    if t == seq_len:
        return seq
    if t > seq_len:
        idx = np.linspace(0, t - 1, seq_len).round().astype(int)
        return seq[idx]
    pad = np.repeat(seq[-1:], seq_len - t, axis=0)
    return np.concatenate([seq, pad], axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=str(config.DATA_RAW_DIR / "archive__4_.zip"))
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--seq-len", type=int, default=config.SEQ_LEN)
    parser.add_argument("--min-samples", type=int, default=config.MIN_SAMPLES_PER_CLASS)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-active-frames", type=int, default=3)
    args = parser.parse_args()

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Opening {args.zip} ...")
    zf = zipfile.ZipFile(args.zip)

    print(f"Scanning entries for sources: {args.sources} ...")
    samples = discover_samples(zf, set(args.sources))
    print(f"Found {len(samples)} raw samples.")

    # First pass: count samples per class, drop junk/low-count classes
    # BEFORE doing the expensive frame loading.
    raw_counts = defaultdict(int)
    for (_, cls, _) in samples.keys():
        raw_counts[cls] += 1

    kept_classes = sorted(
        c for c, n in raw_counts.items()
        if "test" not in c.lower() and n >= args.min_samples
    )
    dropped_classes = sorted(set(raw_counts) - set(kept_classes))
    print(f"Keeping {len(kept_classes)} classes, dropping {len(dropped_classes)} "
          f"(junk or < {args.min_samples} samples): {dropped_classes}")

    class_to_idx = {c: i for i, c in enumerate(kept_classes)}

    X, y, meta = [], [], []
    dropped_bad_recordings = 0
    kept_class_set = set(kept_classes)
    items = [item for item in samples.items() if item[0][1] in kept_class_set]

    for i, ((source, cls, sample_id), frame_map) in enumerate(items):
        seq = load_sequence(zf, frame_map)
        seq = trim_dead_frames(seq)

        active_frames = np.any(np.abs(seq) > 1e-8, axis=1).sum()
        if active_frames < args.min_active_frames:
            dropped_bad_recordings += 1
            continue

        seq = pad_or_truncate(seq, args.seq_len)
        X.append(seq)
        y.append(class_to_idx[cls])
        meta.append({"source": source, "class": cls, "sample_id": sample_id})

        if (i + 1) % 2000 == 0:
            print(f"  processed {i + 1}/{len(items)} samples...")

    zf.close()

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"\nFinal dataset: X={X.shape}, y={y.shape}")
    print(f"Dropped {dropped_bad_recordings} bad/empty recordings.")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    val_fraction_of_temp = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, random_state=args.seed, stratify=y_temp
    )

    print(f"Split sizes: train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

    np.save(config.DATA_PROCESSED_DIR / "X_train.npy", X_train)
    np.save(config.DATA_PROCESSED_DIR / "y_train.npy", y_train)
    np.save(config.DATA_PROCESSED_DIR / "X_val.npy", X_val)
    np.save(config.DATA_PROCESSED_DIR / "y_val.npy", y_val)
    np.save(config.DATA_PROCESSED_DIR / "X_test.npy", X_test)
    np.save(config.DATA_PROCESSED_DIR / "y_test.npy", y_test)

    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump({"idx_to_class": kept_classes, "class_to_idx": class_to_idx}, f, indent=2)

    print(f"\nSaved processed data + label_map.json to: {config.DATA_PROCESSED_DIR}")


if __name__ == "__main__":
    main()
