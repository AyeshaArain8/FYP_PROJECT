"""Safely integrate recorded examples, train, rebuild templates, and evaluate."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


TRACKED_FILES = [
    "data/processed/X_train.npy", "data/processed/y_train.npy",
    "data/processed/X_val.npy", "data/processed/y_val.npy",
    "data/processed/X_test.npy", "data/processed/y_test.npy",
    "data/processed/label_map.json", "data/processed/templates.npz",
    "outputs/checkpoints/psl_model.h5",
]


def run_step(title, command):
    print(f"\n=== {title} ===", flush=True)
    result = subprocess.run(command, cwd=config.ROOT_DIR, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.stdout:
        print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(f"{title} failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word", required=True)
    parser.add_argument("--minimum", type=int, default=20)
    args = parser.parse_args()
    word = re.sub(r"\s+", "_", args.word.strip().lower())
    if not re.fullmatch(r"[a-z0-9]+(?:[_-][a-z0-9]+)*", word):
        parser.error("--word may contain letters, numbers, '_' or '-' only")

    sample_dir = config.DATA_RAW_DIR / "custom_samples" / word
    sample_files = sorted(sample_dir.glob("sample_*.npy")) if sample_dir.exists() else []
    with open(config.LABEL_MAP_PATH, encoding="utf-8") as handle:
        labels = json.load(handle)["idx_to_class"]
    old_classes = list(labels.values()) if isinstance(labels, dict) else list(labels)
    is_new = word not in old_classes
    required = max(args.minimum, 20 if is_new else 7)
    if len(sample_files) < required:
        raise SystemExit(f"'{word}' has {len(sample_files)} recordings; {required} usable samples are required.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = config.OUTPUTS_DIR / "backups" / "word_workflow" / f"{word}_{stamp}"
    for rel in TRACKED_FILES:
        source = config.ROOT_DIR / rel
        target = backup_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    sample_backup = backup_dir / "raw_custom_samples" / word
    sample_backup.mkdir(parents=True, exist_ok=True)
    for path in sample_files:
        shutil.copy2(path, sample_backup / path.name)

    archive_dir = config.DATA_RAW_DIR / "custom_samples_archive" / word
    old_archive_names = {path.name for path in archive_dir.glob("sample_*.npy")} if archive_dir.exists() else set()
    try:
        run_step("Add samples and update labels",
                 [sys.executable, "src/add_recorded_words.py", "--word", word,
                  "--minimum", str(required)])
        if is_new:
            run_step("Expand and train the PSL classifier",
                     [sys.executable, "src/train.py", "--expand-classes"])
        else:
            run_step("Fine-tune the existing PSL class",
                     [sys.executable, "src/train.py", "--resume"])
        run_step("Rebuild Text-to-Sign templates",
                 [sys.executable, "src/build_templates.py"])
        run_step("Evaluate updated model", [sys.executable, "src/evaluate.py"])
    except Exception:
        print("\nTraining failed. Restoring the prior model and dataset snapshot.", flush=True)
        for rel in TRACKED_FILES:
            saved = backup_dir / rel
            shutil.copy2(saved, config.ROOT_DIR / rel)
        sample_dir.mkdir(parents=True, exist_ok=True)
        for saved in sample_backup.glob("sample_*.npy"):
            shutil.copy2(saved, sample_dir / saved.name)
        if archive_dir.exists():
            for archived in archive_dir.glob("sample_*.npy"):
                if archived.name not in old_archive_names:
                    archived.unlink()
        raise

    print(f"\nSuccessfully updated '{word}'. Safety snapshot: {backup_dir}", flush=True)


if __name__ == "__main__":
    main()
