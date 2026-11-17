"""Render landmark templates as a smooth, human-shaped 2D hand animation.

The project contains landmark sequences rather than filmed signing videos.
This renderer keeps the learned movement but draws a warm, shaded hand around
the landmarks instead of displaying the bare MediaPipe stick figure.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

CANVAS = 480
SUPERSAMPLE = 2
SKIN = (218, 153, 119, 255)
SKIN_LIGHT = (244, 192, 158, 255)
SKIN_DARK = (139, 81, 65, 255)
SKIN_OUTLINE = (111, 64, 54, 255)


def load_templates():
    data = np.load(config.TEMPLATES_PATH)
    return {key: data[key] for key in data.files}


def _active_hand(flat_hand):
    points = np.asarray(flat_hand, dtype=np.float32).reshape(21, 3)
    active = np.any(np.abs(points) > 1e-8, axis=1)
    return points, active


def _word_bounds(sequence):
    visible = []
    for frame in sequence:
        for slot in (frame[:63], frame[63:]):
            points, active = _active_hand(slot)
            if active.sum() >= 8:
                visible.append(points[active, :2])
    if not visible:
        return None
    points = np.concatenate(visible, axis=0)
    low = points.min(axis=0)
    high = points.max(axis=0)
    # Keep degenerate coordinates safe and add room for the palm outline.
    span = np.maximum(high - low, 0.04)
    return low, high, span


def _map_hand(flat_hand, bounds, size):
    low, high, span = bounds
    pad = 76 * SUPERSAMPLE
    width = size - 2 * pad
    height = size - 2 * pad - 42 * SUPERSAMPLE
    scale = min(width / span[0], height / span[1])
    content_w, content_h = span * scale
    left = (size - content_w) / 2
    top = pad + (height - content_h) / 2

    points, active = _active_hand(flat_hand)
    mapped = np.zeros((21, 2), dtype=np.float32)
    mapped[:, 0] = left + (points[:, 0] - low[0]) * scale
    mapped[:, 1] = top + (points[:, 1] - low[1]) * scale
    return mapped, active, max(6 * SUPERSAMPLE, span[0] * scale)


def _draw_rounded_path(draw, path, width, *, round_start=False, round_end=True):
    """Draw a soft outlined finger path with rounded joints."""
    if len(path) < 2:
        return
    draw.line(path, fill=SKIN_OUTLINE, width=max(2, width + 5 * SUPERSAMPLE), joint="curve")
    draw.line(path, fill=SKIN_DARK, width=max(2, width + 2 * SUPERSAMPLE), joint="curve")
    draw.line(path, fill=SKIN, width=max(2, width), joint="curve")
    # Blend the base into the palm and give each fingertip a rounded end.
    radius = width * 0.5
    for x, y in path[1:-1]:
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=SKIN)
    endpoints = []
    if round_start:
        endpoints.append((path[0], False))
    if round_end:
        endpoints.append((path[-1], True))
    for (x, y), outlined in endpoints:
        if outlined:
            draw.ellipse((x-radius-2, y-radius-2, x+radius+2, y+radius+2), fill=SKIN_OUTLINE)
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=SKIN)

    # Soft reflected light along one side of each finger gives it a rounded form.
    points = np.asarray(path, dtype=np.float32)
    for first, second in zip(points[:-1], points[1:]):
        direction = second - first
        length = np.linalg.norm(direction)
        if length <= 1:
            continue
        normal = np.asarray([-direction[1], direction[0]]) / length
        if normal[0] > 0:
            normal = -normal
        offset = normal * width * 0.18
        draw.line([tuple(first + offset), tuple(second + offset)],
                  fill=(255, 221, 194, 125), width=max(1, int(width * 0.12)))


def _smooth_polygon(points, corner=0.18, steps=5):
    """Round the landmark palm outline with short quadratic corner curves."""
    points = np.asarray(points, dtype=np.float32)
    smooth = []
    for i, current in enumerate(points):
        previous = points[(i - 1) % len(points)]
        following = points[(i + 1) % len(points)]
        start = current + (previous - current) * corner
        end = current + (following - current) * corner
        for t in np.linspace(0.0, 1.0, steps, endpoint=False):
            curve = ((1-t) ** 2 * start + 2 * (1-t) * t * current + t ** 2 * end)
            smooth.append(tuple(curve))
        smooth.append(tuple(end))
    return smooth


def _draw_hand(image, flat_hand, bounds):
    mapped, active, span_px = _map_hand(flat_hand, bounds, image.width)
    if active.sum() < 8:
        return

    # Landmark template data has all 21 points for a visible hand. Draw a
    # narrow wrist and palm first, then rounded fingers over the palm.
    palm_center = mapped[[5, 9, 13, 17]].mean(axis=0)
    palm_width_px = float(np.linalg.norm(mapped[5] - mapped[17])) if active[5] and active[17] else span_px * 0.42
    palm_width_px = max(18 * SUPERSAMPLE, palm_width_px)
    side_axis = mapped[17] - mapped[5]
    side_length = np.linalg.norm(side_axis)
    side_axis = side_axis / side_length if side_length > 1 else np.asarray([1.0, 0.0])
    # Add gentle outward curves on both palm edges instead of a sharp triangle.
    side5 = mapped[0] + (mapped[5] - mapped[0]) * 0.55 - side_axis * palm_width_px * 0.10
    side17 = mapped[0] + (mapped[17] - mapped[0]) * 0.55 + side_axis * palm_width_px * 0.10
    palm = [tuple(mapped[i]) for i in (0, 1, 2) if active[i]]
    if active[5]:
        palm.extend([tuple(side5), tuple(mapped[5])])
    for index in (9, 13, 17):
        if active[index]:
            palm.append(tuple(mapped[index]))
    if active[17]:
        palm.append(tuple(side17))
    if len(palm) >= 5:
        palm = _smooth_polygon(palm)
        shadow = [(x + 5 * SUPERSAMPLE, y + 7 * SUPERSAMPLE) for x, y in palm]
        draw = ImageDraw.Draw(image, "RGBA")
        draw.polygon(shadow, fill=(71, 50, 45, 48))
        draw.polygon(palm, fill=SKIN_OUTLINE)
        center = np.asarray(palm, dtype=np.float32).mean(axis=0)
        inner = [tuple(center + (np.asarray(point) - center) * 0.965) for point in palm]
        draw.polygon(inner, fill=SKIN)

    draw = ImageDraw.Draw(image, "RGBA")
    palm_width = palm_width_px
    finger_width = max(10 * SUPERSAMPLE, int(palm_width * 0.18))

    # Extend the wrist slightly below landmark zero for a more natural silhouette.
    if active[0] and active[9]:
        wrist = mapped[0]
        away = wrist - mapped[9]
        length = np.linalg.norm(away)
        if length > 1:
            end = wrist + away / length * (palm_width * 0.38)
            _draw_rounded_path(draw, [tuple(end), tuple(wrist)], int(palm_width * 0.34),
                               round_start=False, round_end=False)

    digits = [([1, 2, 3, 4], 0.76), ([5, 6, 7, 8], 1.0),
              ([9, 10, 11, 12], 1.02), ([13, 14, 15, 16], 0.96),
              ([17, 18, 19, 20], 0.84)]
    for ids, width_factor in digits:
        ids = [index for index in ids if active[index]]
        if len(ids) < 2:
            continue
        path = [tuple(mapped[index]) for index in ids]
        width = max(8 * SUPERSAMPLE, int(finger_width * width_factor))
        _draw_rounded_path(draw, path, width, round_start=False, round_end=True)

        # A small, softly lit nail makes the fingertip read as a hand rather
        # than a connected tube. Its orientation follows the last finger joint.
        tip = np.asarray(path[-1])
        before = np.asarray(path[-2])
        direction = tip - before
        length = np.linalg.norm(direction)
        if length > 1:
            center = tip - direction / length * (width * 0.28)
            radius_x, radius_y = width * 0.19, width * 0.12
            draw.ellipse((center[0]-radius_x, center[1]-radius_y,
                          center[0]+radius_x, center[1]+radius_y),
                         fill=SKIN_LIGHT, outline=(205, 145, 125, 210),
                         width=max(1, SUPERSAMPLE))

    # Gentle palm creases and highlights add depth without covering the sign.
    if all(active[i] for i in (0, 5, 9, 13, 17)):
        crease = (155, 96, 76, 105)
        draw.line([tuple(mapped[5]), tuple(mapped[9]), tuple(mapped[13])],
                  fill=crease, width=1 * SUPERSAMPLE, joint="curve")


def _font(size):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _render_frame(frame, bounds, word):
    size = CANVAS * SUPERSAMPLE
    image = Image.new("RGBA", (size, size), (250, 251, 253, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((12*SUPERSAMPLE, 12*SUPERSAMPLE,
                            size-12*SUPERSAMPLE, size-12*SUPERSAMPLE),
                           radius=20*SUPERSAMPLE,
                           outline=(225, 229, 236, 255), width=2*SUPERSAMPLE)
    font = _font(22 * SUPERSAMPLE)
    label = word.replace("_", " ").replace("-", " ").title()
    draw.text((size//2, 38*SUPERSAMPLE), label, anchor="mm",
              font=font, fill=(48, 55, 68, 255))

    for slot in (frame[:63], frame[63:]):
        _draw_hand(image, slot, bounds)

    return image.convert("RGB").resize((CANVAS, CANVAS), Image.Resampling.LANCZOS)


def render_sign_sequence(words: list, out_path: str = None):
    if out_path is None:
        out_path = str(config.PLOTS_DIR / "text_to_sign.gif")

    templates = load_templates()
    missing = [word for word in words if word not in templates]
    if missing:
        print(f"Warning: no sign template for: {missing} (skipping these words)")
    words = [word for word in words if word in templates]
    if not words:
        print("No matching signs found for this text -- nothing to render.")
        return None

    frames = []
    for word in words:
        sequence = templates[word]
        bounds = _word_bounds(sequence)
        if bounds is None:
            continue
        for frame in sequence:
            frames.append(_render_frame(frame, bounds, word))
    if not frames:
        return None

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paletted = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for frame in frames]
    paletted[0].save(out_path, save_all=True, append_images=paletted[1:],
                     duration=50, loop=0, disposal=2, optimize=False)
    print(f"Saved sign animation to {out_path}")
    return out_path


if __name__ == "__main__":
    render_sign_sequence(sys.argv[1:])
