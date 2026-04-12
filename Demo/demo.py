import cv2
import time
import numpy as np

from Pipeline.driver import Pipeline


# ---------------- CONFIG ---------------- #
VIDEO_PATH = "test.mp4"

WINDOW_NAME = "Saber Action Demo"

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------- DRAWING ---------------- #

def draw_text(img, text, x, y, scale=0.5, thickness=1):
    cv2.putText(img, text, (x, y), FONT, scale, (0, 255, 0), thickness, cv2.LINE_AA)


def draw_roi(frame, roi):
    if roi is None:
        return
    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)


def draw_keypoints(frame, kpts):
    if kpts is None or len(kpts) == 0:
        return

    for x, y in kpts:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), -1)


def draw_topk(frame, topk, x, y):
    for i, entry in enumerate(topk):
        text = f"{entry['label']}: {entry['confidence']:.2f}"
        draw_text(frame, text, x, y + i * 20)


def draw_timing(frame, timing, fps):
    draw_text(frame, f"FPS: {fps:.2f}", 20, 100)

    draw_text(frame, f"Total: {timing['total']:.1f} ms", 20, 130)
    draw_text(frame, f"Pose: {timing['pose']:.1f} ms", 20, 150)
    draw_text(frame, f"Cls: {timing['classify']:.1f} ms", 20, 170)


# ---------------- MAIN LOOP ---------------- #

def run_demo():
    pipe = Pipeline()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")

    paused = False
    fps = 0.0

    prev_time = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            result = pipe.step(frame)

            # -------- DRAW -------- #
            draw_roi(frame, result["roi"])

            draw_keypoints(frame, result["left"]["kpts"])
            draw_keypoints(frame, result["right"]["kpts"])

            draw_topk(frame, result["left"]["topk"], 20, 30)
            draw_topk(frame, result["right"]["topk"], 20, 120)
            draw_timing(frame, result["timing"], fps)

            # -------- FPS -------- #
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):  # pause
            paused = not paused
        elif key == ord("s"):  # step frame
            paused = True
            ret, frame = cap.read()
            if ret:
                result = pipe.step(frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo()