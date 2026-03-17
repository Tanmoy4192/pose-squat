import mediapipe as mp
import cv2
import time


class ReferenceAnalyzer:

    def __init__(self, pose_engine):

        self.pose_engine = pose_engine
        self.reference_landmarks = None

    def extract(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(time.time() * 1000)

        self.pose_engine.detect_async(mp_image, timestamp)

        if self.pose_engine.latest_result and self.pose_engine.latest_result.pose_landmarks:
            self.reference_landmarks = self.pose_engine.latest_result.pose_landmarks[0]

        return self.reference_landmarks