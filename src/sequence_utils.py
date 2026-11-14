"""Sequence preprocessing shared by live inference and custom recordings."""

import numpy as np


def trim_dead_frames(seq: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    active = np.any(np.abs(seq) > tolerance, axis=1)
    if not active.any():
        return seq[:0]
    first = int(np.argmax(active))
    last = len(active) - int(np.argmax(active[::-1]))
    return seq[first:last]


def pad_or_truncate(seq: np.ndarray, seq_len: int) -> np.ndarray:
    if len(seq) == 0:
        return np.zeros((seq_len, 126), dtype=np.float32)
    if len(seq) == seq_len:
        return seq.astype(np.float32, copy=False)
    if len(seq) > seq_len:
        indices = np.linspace(0, len(seq) - 1, seq_len).round().astype(int)
        return seq[indices].astype(np.float32, copy=False)
    padding = np.repeat(seq[-1:], seq_len - len(seq), axis=0)
    return np.concatenate((seq, padding), axis=0).astype(np.float32, copy=False)


def prepare_live_sequence(seq: np.ndarray, seq_len: int, min_active_frames: int = 3):
    trimmed = trim_dead_frames(np.asarray(seq, dtype=np.float32))
    active_count = int(np.any(np.abs(trimmed) > 1e-8, axis=1).sum())
    if active_count < min_active_frames:
        return None
    return pad_or_truncate(trimmed, seq_len)
