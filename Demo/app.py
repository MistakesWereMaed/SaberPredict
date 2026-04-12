import streamlit as st
import cv2
import time
import numpy as np

from Pipeline.driver import Pipeline


# ---------------- CONFIG ---------------- #
VIDEO_PATHS = {
    "Demo Clip": "test.mp4",
    # add more here
}

FRAME_DELAY = 0.03  # ~30 FPS


# ---------------- DRAWING ---------------- #

def draw_text(img, text, x, y, scale=0.5, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 255, 0), thickness, cv2.LINE_AA)


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
    if topk is None:
        return
    for i, entry in enumerate(topk):
        text = f"{entry['label']}: {entry['confidence']:.2f}"
        draw_text(frame, text, x, y + i * 20)


def draw_timing(frame, timing, fps):
    draw_text(frame, f"FPS: {fps:.2f}", 20, 100)
    draw_text(frame, f"Total: {timing['total']:.1f} ms", 20, 130)
    draw_text(frame, f"Pose: {timing['pose']:.1f} ms", 20, 150)
    draw_text(frame, f"Cls: {timing['classify']:.1f} ms", 20, 170)


# ---------------- SESSION INIT ---------------- #

if "pipeline" not in st.session_state:
    st.session_state.pipeline = Pipeline()

if "cap" not in st.session_state:
    st.session_state.cap = None

if "playing" not in st.session_state:
    st.session_state.playing = False

if "frame" not in st.session_state:
    st.session_state.frame = None

if "prev_time" not in st.session_state:
    st.session_state.prev_time = time.time()

if "fps" not in st.session_state:
    st.session_state.fps = 0.0


# ---------------- UI ---------------- #

st.title("Saber Action Classification Demo")

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("Controls")

    video_name = st.selectbox("Select Video", list(VIDEO_PATHS.keys()))
    video_path = VIDEO_PATHS[video_name]

    if st.button("Load Video"):
        st.session_state.cap = cv2.VideoCapture(video_path)
        st.session_state.pipeline = Pipeline()  # reset buffers
        st.session_state.playing = False

    if st.button("Play"):
        st.session_state.playing = True

    if st.button("Pause"):
        st.session_state.playing = False

    if st.button("Step"):
        st.session_state.playing = False
        if st.session_state.cap:
            ret, frame = st.session_state.cap.read()
            if ret:
                st.session_state.frame = frame

    st.markdown("---")

    st.subheader("Metrics")
    fps_placeholder = st.empty()
    timing_placeholder = st.empty()


with col1:
    frame_placeholder = st.empty()


# ---------------- MAIN UPDATE ---------------- #

def process_frame():
    cap = st.session_state.cap
    pipe = st.session_state.pipeline

    if cap is None:
        return None

    ret, frame = cap.read()
    if not ret:
        return None

    result = pipe.step(frame)

    # -------- DRAW -------- #
    draw_roi(frame, result["roi"])
    draw_keypoints(frame, result["left"]["kpts"])
    draw_keypoints(frame, result["right"]["kpts"])

    draw_topk(frame, result["left"]["topk"], 20, 30)
    draw_topk(frame, result["right"]["topk"], 20, 120)

    # -------- FPS -------- #
    curr_time = time.time()
    fps = 1.0 / (curr_time - st.session_state.prev_time)
    st.session_state.prev_time = curr_time
    st.session_state.fps = fps

    draw_timing(frame, result["timing"], fps)

    # update metrics panel
    fps_placeholder.write(f"FPS: {fps:.2f}")
    timing_placeholder.write(result["timing"])

    return frame


# ---------------- RENDER LOOP ---------------- #

if st.session_state.playing:
    frame = process_frame()
    if frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

    time.sleep(FRAME_DELAY)
    st.rerun()

else:
    if st.session_state.frame is not None:
        frame = process_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")