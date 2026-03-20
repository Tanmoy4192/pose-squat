import os
import cv2
import numpy as np
import mediapipe as mp
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response

from pose_engine import PoseEngine
from video_controller import ReferenceVideo
from reference_analyzer import ReferenceAnalyzer
from squat_analyzer import WorkoutController
from ui_renderer import (
    draw_alert, draw_rep_counter,
    draw_phase_indicator, draw_similarity_score
)

app = FastAPI()

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "models", "pose_landmarker_full.task")
REF_VIDEO_PATH = os.path.join(BASE_DIR, "reference", "squat.mp4")

# These are created once and reused every request
user_detector      = PoseEngine(MODEL_PATH)
reference_video    = ReferenceVideo(REF_VIDEO_PATH)
reference_analyzer = ReferenceAnalyzer(MODEL_PATH)
controller         = WorkoutController()
last_similarity    = 0.0


def letterbox(frame, target_w, target_h):
    src_h, src_w = frame.shape[:2]
    scale        = min(target_w / src_w, target_h / src_h)
    new_w        = int(src_w * scale)
    new_h        = int(src_h * scale)
    resized      = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas       = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off        = (target_w - new_w) // 2
    y_off        = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


@app.post("/process")
async def process_frame(file: UploadFile = File(...)):
    """
    Browser sends a webcam frame here as an image file.
    We run pose detection on it and send back the processed frame.
    """
    global last_similarity

    # Read the image bytes sent from browser
    contents = await file.read()
    np_arr   = np.frombuffer(contents, np.uint8)
    frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return Response(content=b"", media_type="image/jpeg")

    height, width, _ = frame.shape

    # Run pose detection
    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp = int(time.time() * 1000)
    user_detector.detect_async(mp_image, timestamp)

    user_result    = user_detector.get_latest_result()
    user_landmarks = None
    if user_result and user_result.pose_landmarks:
        user_landmarks = user_result.pose_landmarks[0]

    # Reference frame
    ref_frame = reference_video.read()
    if ref_frame is None:
        return Response(content=b"", media_type="image/jpeg")
    ref_frame = letterbox(ref_frame, width, height)
    ref_landmarks = reference_analyzer.extract(ref_frame)

    # Evaluate form
    advance_video, message, correct = controller.evaluate(
        user_landmarks, ref_landmarks, width, height
    )

    if advance_video:
        reference_video.resume()
    else:
        reference_video.pause()

    # Draw on frames
    frame = user_detector.draw_skeleton(frame, correct)
    draw_phase_indicator(frame, controller.phase)

    buf = controller._smoother._buffer
    if buf:
        last_similarity = sum(buf) / len(buf)
    draw_similarity_score(frame, last_similarity)

    draw_alert(ref_frame, message)
    draw_rep_counter(ref_frame, controller.rep_count)

    # Combine side by side
    combined = cv2.hconcat([frame, ref_frame])

    # Encode and send back as JPEG
    _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/")
def index():
    with open("static/index.html") as f:
        return HTMLResponse(f.read())