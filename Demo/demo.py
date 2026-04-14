import cv2
import pickle
import numpy as np

CACHE_PATHS = [
    "Demo/Cache/video_0.pkl",
    "Demo/Cache/video_1.pkl",
    "Demo/Cache/video_2.pkl",
]

WINDOW = "Saber Demo"
FONT = cv2.FONT_HERSHEY_SIMPLEX

EDGES = [
    (0, 1), (1, 3),
    (3, 5), (1, 2),
    (0, 2), (2, 4),
    (4, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (11, 12),
    (5, 11), (6, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]


# ---------------- IO ---------------- #

def load_cache(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------- DRAW HELPERS ---------------- #

def text(img, s, x, y, color=(0, 255, 0), scale=0.7, thickness=2):
    # Get text size
    (tw, th), baseline = cv2.getTextSize(
        s,
        FONT,
        scale,
        thickness
    )

    # Background rectangle coordinates
    x1, y1 = x, y - th - baseline
    x2, y2 = x + tw, y + baseline

    # Clamp to image bounds (prevents crashes near edges)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    # Draw black background box
    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        (0, 0, 0),
        -1
    )

    # Draw text on top
    cv2.putText(
        img,
        s,
        (x, y),
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def skeleton(img, kpts, color):
    if kpts is None:
        return

    kpts = np.array(kpts)

    for i, j in EDGES:
        if i < len(kpts) and j < len(kpts):
            x1, y1 = kpts[i]
            x2, y2 = kpts[j]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                cv2.line(img,
                         (int(x1), int(y1)),
                         (int(x2), int(y2)),
                         color, 2, cv2.LINE_AA)


def roi_box(img, roi):
    if roi is None:
        return
    x1, y1, x2, y2 = roi
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)


# ---------------- OVERLAYS ---------------- #

def draw_perf(img, t, fps):
    y = 30
    x = 20
    spacing = 32

    text(img, f"FPS: {fps:.1f}", x, y); y += spacing
    text(img, f"Total: {t.get('total', 0):.1f} ms", x, y); y += spacing
    text(img, f"ROI: {t.get('roi', 0):.1f} ms", x, y); y += spacing
    text(img, f"Pose: {t.get('pose', 0):.1f} ms", x, y); y += spacing
    text(img, f"Filter: {t.get('filter', 0):.1f} ms", x, y); y += spacing
    text(img, f"Cls: {t.get('classify', 0):.1f} ms", x, y)


def draw_controls(img):
    h, w = img.shape[:2]
    x = w - 420
    y = 30

    for l in [
        "[SPACE] pause",
        "[F] forward",
        "[D] back",
        "[R] reset",
        "[1-3] switch",
        "[Q] quit"
    ]:
        text(img, l, x, y, (255, 255, 255))
        y += 32


# ---------------- LABEL + TOPK ---------------- #

def draw_fencer_info(img, res, side):
    data = res.get(side, {})
    kpts = data.get("kpts", None)

    if kpts is None or len(kpts) == 0:
        return

    # anchor: torso or head (joint 0 fallback)
    x, y = kpts[0]

    true_label = data.get("true_label", None)
    pred_label = data.get("label", None)

    color = (0, 0, 255) if side == "left" else (0, 255, 0)

    # main label
    label = f"PD | {pred_label}"
    text(img, label, int(x) - 40, int(y) - 40, color)

    # ground truth (if available)
    if true_label is not None:
        text(img, f"GT | {true_label}", int(x) - 40, int(y) - 70, color)

    # TOP-K
    topk = data.get("topk", None)
    x_offset = -300 if side == "left" else 200
    if topk:
        for i, t in enumerate(topk[:3]):
            s = f"{t['label']}: {t['confidence']:.2f}"
            text(img, s, int(x) + x_offset, int(y) + 20 + i * 25, (255, 255, 255), scale=0.7)


# ---------------- MAIN LOOP ---------------- #

def run(cache_path):
    cache = load_cache(cache_path)

    frames = cache["frames"]
    results = cache["results"]

    idx = 0
    paused = False

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    running = True

    while running:

        if len(frames) == 0:
            break

        frame = frames[idx].copy()
        res = results[idx]

        t = res.get("timing", {})

        total_ms = max(t.get("total", 1e-6), 1e-6)
        fps = 1000.0 / total_ms

        # ---------------- DRAW ---------------- #
        roi_box(frame, res.get("roi"))

        skeleton(frame, res.get("left", {}).get("kpts"), (0, 0, 255))
        skeleton(frame, res.get("right", {}).get("kpts"), (0, 255, 0))

        draw_fencer_info(frame, res, "left")
        draw_fencer_info(frame, res, "right")

        draw_perf(frame, t, fps)
        draw_controls(frame)

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKey(30) & 0xFF

        # ---------------- CONTROLS ---------------- #
        if key == ord('q'):
            running = False

        elif key == ord('f'):
            idx = min(idx + 1, len(frames) - 1)

        elif key == ord('d'):
            idx = max(idx - 1, 0)

        elif key == ord('r'):
            idx = 0

        elif key == ord(' '):
            paused = not paused

        elif key in [ord('1'), ord('2'), ord('3')]:
            i = key - ord('1')
            if i < len(CACHE_PATHS):
                cache = load_cache(CACHE_PATHS[i])
                frames = cache["frames"]
                results = cache["results"]
                idx = 0

        if not paused:
            idx = min(idx + 1, len(frames) - 1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run(CACHE_PATHS[0])