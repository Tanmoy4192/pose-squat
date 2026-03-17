import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

POSE_CONNECTIONS = [

    (11,12),

    (11,13),(13,15),
    (12,14),(14,16),

    (23,25),(25,27),
    (24,26),(26,28)
]

VISIBLE_LANDMARKS = [

    11,12,
    13,14,
    15,16,
    23,24,
    25,26,
    27,28
]

class PoseEngine:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            result_callback=self._callback
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.latest_result = None

    def _callback(self, result, image, timestamp):
        self.latest_result = result

    def detect_async(self, mp_image, timestamp):
        self.landmarker.detect_async(mp_image, timestamp)

    def draw_v_bone(self, frame, p1, p2, color):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        length = np.hypot(dx, dy)
        if length == 0:
            return

        nx = -dy / length
        ny = dx / length

        spread = 6

        p1a = (int(p1[0] + nx * spread), int(p1[1] + ny * spread))
        p1b = (int(p1[0] - nx * spread), int(p1[1] - ny * spread))

        cv2.line(frame, p1a, p2, color, 2, cv2.LINE_AA)
        cv2.line(frame, p1b, p2, color, 2, cv2.LINE_AA)


    def draw_skeleton(self, frame, correct=True):
        if self.latest_result is None:
            return frame

        if not self.latest_result.pose_landmarks:
            return frame

        height, width, _ = frame.shape
        landmarks = self.latest_result.pose_landmarks[0]

        color = (240,240,240) if correct else (0,0,255)

        points = {}

        for idx in VISIBLE_LANDMARKS:
            lm = landmarks[idx]
            x = int(lm.x * width)
            y = int(lm.y * height)

            points[idx] = (x,y)

        # LIMBS
        for start,end in POSE_CONNECTIONS:

            if start in points and end in points:

                self.draw_v_bone(
                    frame,
                    points[start],
                    points[end],
                    color
                )

        # SPINE

        left_shoulder = points[11]
        right_shoulder = points[12]

        left_hip = points[23]
        right_hip = points[24]

        shoulder_mid = (
            int((left_shoulder[0] + right_shoulder[0]) / 2),
            int((left_shoulder[1] + right_shoulder[1]) / 2)
        )

        hip_mid = (
            int((left_hip[0] + right_hip[0]) / 2),
            int((left_hip[1] + right_hip[1]) / 2)
        )

        # create intermediate spine joints
        upper_spine = (
            int((shoulder_mid[0]*3 + hip_mid[0]) / 4),
            int((shoulder_mid[1]*3 + hip_mid[1]) / 4)
        )

        mid_spine = (
            int((shoulder_mid[0] + hip_mid[0]) / 2),
            int((shoulder_mid[1] + hip_mid[1]) / 2)
        )

        lower_spine = (
            int((shoulder_mid[0] + hip_mid[0]*3) / 4),
            int((shoulder_mid[1] + hip_mid[1]*3) / 4)
        )

        spine_points = [
            shoulder_mid,
            upper_spine,
            mid_spine,
            lower_spine,
            hip_mid
        ]

        # draw spine bones
        for i in range(len(spine_points)-1):

            self.draw_v_bone(
                frame,
                spine_points[i],
                spine_points[i+1],
                color
            )

        # draw spine joints
        for p in spine_points:
            cv2.circle(
                frame,
                p,
                7,
                color,
                -1,
                lineType=cv2.LINE_AA
            )

        # DRAW NORMAL JOINTS
        for p in points.values():
            cv2.circle(
                frame,
                p,
                7,
                color,
                -1,
                lineType=cv2.LINE_AA
            )

        return frame