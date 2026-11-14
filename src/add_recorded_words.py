"""Add recorded custom words to the data splits before retraining.

Run: python src/add_recorded_words.py
Then run train.py, build_templates.py, and evaluate.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from sequence_utils import prepare_live_sequence


def ordered_classes(label_map):
    values = label_map["idx_to_class"]
    if isinstance(values, dict):
        return [name for _, name in sorted(values.items(), key=lambda item: int(item[0]))]
    return list(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum", type=int, default=7,
                        help="minimum usable recordings for calibration; new classes still require 20")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sample_root = config.DATA_RAW_DIR / "custom_samples"
    if not sample_root.exists():
        parser.error(f"No recordings found under {sample_root}; run record_word.py first")

    with open(config.LABEL_MAP_PATH, encoding="utf-8") as f:
        label_map = json.load(f)
    classes = ordered_classes(label_map)
    splits = {
        name: [np.load(config.DATA_PROCESSED_DIR / f"X_{name}.npy"),
               np.load(config.DATA_PROCESSED_DIR / f"y_{name}.npy")]
        for name in ("train", "val", "test")
    }
    rng = np.random.default_rng(args.seed)
    updated_words = []
    processed_dirs = []

    for word_dir in sorted(p for p in sample_root.iterdir() if p.is_dir()):
        word = word_dir.name.lower()
        is_new_word = word not in classes
        sequences = []
        for path in sorted(word_dir.glob("sample_*.npy")):
            sequence = prepare_live_sequence(np.load(path), config.SEQ_LEN)
            if sequence is not None:
                sequences.append(sequence)
        required = max(args.minimum, 20 if is_new_word else 1)
        if len(sequences) < required:
            print(f"Skipping '{word}': {len(sequences)} usable samples; need {required}.")
            continue

        if is_new_word:
            class_index = len(classes)
            classes.append(word)
        else:
            class_index = classes.index(word)
        updated_words.append(word)
        processed_dirs.append(word_dir)
        sequences = np.asarray(sequences, dtype=np.float32)
        order = rng.permutation(len(sequences))
        if is_new_word:
            n_test = max(1, round(0.15 * len(order)))
            n_val = max(1, round(0.15 * len(order)))
            split_ids = {
                "test": order[:n_test],
                "val": order[n_test:n_test + n_val],
                "train": order[n_test + n_val:],
            }
        else:
            # Keep the original held-out test set untouched for known words.
            n_val = max(1, round(0.2 * len(order)))
            split_ids = {
                "train": order[n_val:],
                "val": order[:n_val],
                "test": np.array([], dtype=int),
            }
        for split_name, ids in split_ids.items():
            X, y = splits[split_name]
            splits[split_name] = [
                np.concatenate((X, sequences[ids]), axis=0),
                np.concatenate((y, np.full(len(ids), class_index, dtype=y.dtype))),
            ]
        action = "Added new class" if is_new_word else "Added calibration samples"
        print(f"{action} '{word}': train {len(split_ids['train'])}, val {len(split_ids['val'])}, test {len(split_ids['test'])}.")

    if not updated_words:
        print("No usable new or calibration samples were found.")
        return

    for name, (X, y) in splits.items():
        np.save(config.DATA_PROCESSED_DIR / f"X_{name}.npy", X)
        np.save(config.DATA_PROCESSED_DIR / f"y_{name}.npy", y)
    updated = {"idx_to_class": classes,
               "class_to_idx": {name: idx for idx, name in enumerate(classes)}}
    with open(config.LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    archive_root = config.DATA_RAW_DIR / "custom_samples_archive"
    for word_dir in processed_dirs:
        archive_dir = archive_root / word_dir.name
        archive_dir.mkdir(parents=True, exist_ok=True)
        old_ids = [int(p.stem.split("_")[-1]) for p in archive_dir.glob("sample_*.npy")
                   if p.stem.split("_")[-1].isdigit()]
        next_id = max(old_ids, default=0) + 1
        for sample_path in sorted(word_dir.glob("sample_*.npy")):
            destination = archive_dir / f"sample_{next_id:04d}.npy"
            sample_path.replace(destination)
            next_id += 1
    print(f"Updated {len(updated_words)} word(s); vocabulary has {len(classes)} words. Retrain the model and rebuild templates next.")


if __name__ == "__main__":
    main()
