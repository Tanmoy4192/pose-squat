from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import time
import mediapipe as mp

from app.camera import Camera
from app.pose_detector import PoseEngine

app = FastAPI()

camera = Camera()

pose = PoseEngine("models/pose_landmarker_full.task")


def generate_frames():

    while True:

        frame = camera.read()

        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(time.time() * 1000)

        pose.detect_async(mp_image, timestamp)

        frame = pose.draw_skeleton(frame)

        ret, buffer = cv2.imencode(".jpg", frame)

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

@app.get("/video")
def video_feed():

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )