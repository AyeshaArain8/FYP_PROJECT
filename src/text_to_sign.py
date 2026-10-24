"""
Text -> Sign (reverse direction).

There's no filmed video library for every sign, so instead this renders
each word's hand-landmark template (built by build_templates.py) as an
animated stick figure. It reuses the exact landmark data the model was
trained on, so this stays self-contained without needing a separately
filmed video asset library.

USAGE
    python src/text_to_sign.py hello thankyou
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# MediaPipe hand connections (which landmark indices to draw a line between)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def load_templates():
    data = np.load(config.TEMPLATES_PATH)
    return {k: data[k] for k in data.files}


def render_sign_sequence(words: list, out_path: str = None):
    if out_path is None:
        out_path = str(config.PLOTS_DIR / "text_to_sign.gif")

    templates = load_templates()
    missing = [w for w in words if w not in templates]
    if missing:
        print(f"Warning: no sign template for: {missing} (skipping these words)")
    words = [w for w in words if w in templates]
    if not words:
        print("No matching signs found for this text -- nothing to render.")
        return None

    sequence = np.concatenate([templates[w] for w in words], axis=0)  # (T, 126)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # y is flipped: MediaPipe's origin is top-left
    ax.axis("off")

    scat_l, = ax.plot([], [], "o", color="tab:blue", markersize=3)
    scat_r, = ax.plot([], [], "o", color="tab:red", markersize=3)
    lines_l = [ax.plot([], [], color="tab:blue", linewidth=1)[0] for _ in HAND_CONNECTIONS]
    lines_r = [ax.plot([], [], color="tab:red", linewidth=1)[0] for _ in HAND_CONNECTIONS]

    def update(frame_idx):
        frame = sequence[frame_idx]
        left = frame[:63].reshape(21, 3)
        right = frame[63:].reshape(21, 3)
        scat_l.set_data(left[:, 0], left[:, 1])
        scat_r.set_data(right[:, 0], right[:, 1])
        for line, (a, b) in zip(lines_l, HAND_CONNECTIONS):
            line.set_data([left[a, 0], left[b, 0]], [left[a, 1], left[b, 1]])
        for line, (a, b) in zip(lines_r, HAND_CONNECTIONS):
            line.set_data([right[a, 0], right[b, 0]], [right[a, 1], right[b, 1]])
        return [scat_l, scat_r, *lines_l, *lines_r]

    ani = animation.FuncAnimation(fig, update, frames=len(sequence), interval=50, blit=True)
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer="pillow", fps=20)
    plt.close(fig)
    print(f"Saved sign animation to {out_path}")
    return out_path


if __name__ == "__main__":
    render_sign_sequence(sys.argv[1:])
