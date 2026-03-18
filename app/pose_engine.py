import threading
import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

POSE_CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]

VISIBLE_LANDMARKS = [
    11, 12,
    13, 14,
    15, 16,
    23, 24,
    25, 26,
    27, 28,
]

VISIBILITY_THRESHOLD = 0.5


class PoseEngine:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=1,
            result_callback=self._callback,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.latest_result = None
        self._lock = threading.Lock()          
    
    def _callback(self, result, image, timestamp):
        with self._lock:
            self.latest_result = result

    def detect_async(self, mp_image, timestamp):
        self.landmarker.detect_async(mp_image, timestamp)

    def get_latest_result(self):
        """Thread-safe read of the latest detection result."""
        with self._lock:
            return self.latest_result

    # Drawing skeleton
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
        result = self.get_latest_result()
        if result is None or not result.pose_landmarks:
            return frame

        height, width, _ = frame.shape
        landmarks = result.pose_landmarks[0]
        color = (240, 240, 240) if correct else (0, 0, 255)

        # Only include landmarks with sufficient visibility
        points = {}
        for idx in VISIBLE_LANDMARKS:
            lm = landmarks[idx]
            if getattr(lm, "visibility", 1.0) < VISIBILITY_THRESHOLD:
                continue
            points[idx] = (int(lm.x * width), int(lm.y * height))

        # Limbs
        for start, end in POSE_CONNECTIONS:
            if start in points and end in points:
                self.draw_v_bone(frame, points[start], points[end], color)

        # Spine — only draw if all four anchor landmarks are visible
        if all(idx in points for idx in [11, 12, 23, 24]):
            left_shoulder  = points[11]
            right_shoulder = points[12]
            left_hip       = points[23]
            right_hip      = points[24]

            shoulder_mid = (
                (left_shoulder[0] + right_shoulder[0]) // 2,
                (left_shoulder[1] + right_shoulder[1]) // 2,
            )
            hip_mid = (
                (left_hip[0] + right_hip[0]) // 2,
                (left_hip[1] + right_hip[1]) // 2,
            )

            upper_spine = (
                (shoulder_mid[0] * 3 + hip_mid[0]) // 4,
                (shoulder_mid[1] * 3 + hip_mid[1]) // 4,
            )
            mid_spine = (
                (shoulder_mid[0] + hip_mid[0]) // 2,
                (shoulder_mid[1] + hip_mid[1]) // 2,
            )
            lower_spine = (
                (shoulder_mid[0] + hip_mid[0] * 3) // 4,
                (shoulder_mid[1] + hip_mid[1] * 3) // 4,
            )

            spine_points = [shoulder_mid, upper_spine, mid_spine, lower_spine, hip_mid]

            for i in range(len(spine_points) - 1):
                self.draw_v_bone(frame, spine_points[i], spine_points[i + 1], color)

            for p in spine_points:
                cv2.circle(frame, p, 7, color, -1, lineType=cv2.LINE_AA)

        # Joints
        for p in points.values():
            cv2.circle(frame, p, 7, color, -1, lineType=cv2.LINE_AA)

        return frame


# ------------------------------------------------------------------
# Synchronous engine — used by ReferenceAnalyzer (IMAGE mode)
# Keeps reference detection completely separate from live detection
# ------------------------------------------------------------------
class ImagePoseEngine:
    """
    Synchronous single-image pose detector.
    Use this for reference video frames to avoid async timing issues.
    """

    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, mp_image):
        result = self.landmarker.detect(mp_image)
        if result and result.pose_landmarks:
            return result.pose_landmarks[0]
        return None