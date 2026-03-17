from pose_similarity import pose_similarity
from utils import calculate_angle
import math


class WorkoutController:

    def __init__(self):

        self.rep_count = 0
        self.phase = "UP"

        self.similarity_threshold = 0.84

    def evaluate(self, user_lm, ref_lm, width, height):

        if user_lm is None or ref_lm is None:
            return False, "Detecting pose..."

        similarity = pose_similarity(user_lm, ref_lm)

        if similarity < self.similarity_threshold:
            return False, "Follow the mentor pose"

        message = self.check_squat_pose(user_lm, width, height)

        if message:
            return False, message

        self.detect_rep(user_lm, width, height)

        return True, "Good form"

    def check_squat_pose(self, lm, width, height):

        left_hip = (lm[23].x * width, lm[23].y * height)
        left_knee = (lm[25].x * width, lm[25].y * height)
        left_ankle = (lm[27].x * width, lm[27].y * height)

        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        if knee_angle > 120:
            return "Go lower into squat"

        return None

    def detect_rep(self, lm, width, height):

        left_hip = (lm[23].x * width, lm[23].y * height)
        left_knee = (lm[25].x * width, lm[25].y * height)
        left_ankle = (lm[27].x * width, lm[27].y * height)

        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

        squatting = knee_angle < 100

        if squatting and self.phase == "UP":
            self.phase = "DOWN"

        if not squatting and self.phase == "DOWN":
            self.phase = "UP"
            self.rep_count += 1