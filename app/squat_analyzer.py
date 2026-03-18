from collections import Counter
from pose_similarity import SmoothedSimilarity
from utils import calculate_angle, landmarks_visible, get_point

# Thresholds 
SQUAT_DOWN_ANGLE = 100
SQUAT_UP_ANGLE = 155
DEPTH_MIN_ANGLE = 120
TORSO_LEAN_LIMIT = 0.15
STANCE_MIN_RATIO = 0.70
STANCE_MAX_RATIO = 1.70
SYNC_ANGLE_TOLERANCE = 25
VISIBILITY_THRESHOLD = 0.5

# Debounce: a message must appear this many consecutive frames before
# it replaces the currently displayed message.
# Higher = more stable but slightly slower to react.
DEBOUNCE_FRAMES = 8

# Short messages 
MSG_DETECTING = "Detecting..."
MSG_GOOD = "Good form!"
MSG_GO_LOWER = "Go lower"
MSG_BACK = "Back straight"
MSG_WIDEN = "Widen stance"
MSG_NARROW = "Feet closer"
MSG_FOLLOW_DOWN = "Go lower"        # sync cue when user is above ref
MSG_FOLLOW_UP = "Come up"         # sync cue when user is below ref
MSG_FOLLOW = "Follow mentor"

class WorkoutController:
    def __init__(self):
        self.rep_count = 0
        self.phase     = "UP"
        self._smoother = SmoothedSimilarity(window=5)

        # Debounce state
        self._pending_msg = MSG_DETECTING  # candidate message accumulating votes
        self._pending_count = 0             # consecutive frames with pending_msg
        self._stable_msg = MSG_DETECTING  # currently displayed message

    # Public entry point 
    def evaluate(self, user_lm, ref_lm, width, height):
        if user_lm is None or ref_lm is None:
            return False, self._debounce(MSG_DETECTING), False

        user_angle = self._get_knee_angle(user_lm, width, height)
        ref_angle = self._get_knee_angle(ref_lm,  width, height)

        if user_angle is None:
            return False, self._debounce(MSG_DETECTING), False

        # Sync check
        if ref_angle is not None:
            in_sync = abs(user_angle - ref_angle) <= SYNC_ANGLE_TOLERANCE
        else:
            in_sync = True

        # Form check
        form_msg = self._check_form(user_lm, user_angle, ref_angle, width, height)
        correct  = (form_msg is None)

        # Pick raw message this frame
        if not in_sync:
            raw_msg = self._sync_cue(user_angle, ref_angle)
            correct = False
        elif form_msg:
            raw_msg = form_msg
        else:
            raw_msg = MSG_GOOD

        # Rep counting
        self._detect_rep(user_angle)

        advance_video = in_sync and correct
        return advance_video, self._debounce(raw_msg), correct

    # Debounce 
    def _debounce(self, raw_msg):
        if raw_msg == self._pending_msg:
            self._pending_count += 1
        else:
            # New candidate — reset counter
            self._pending_msg   = raw_msg
            self._pending_count = 1

        if self._pending_count >= DEBOUNCE_FRAMES:
            self._stable_msg = self._pending_msg

        return self._stable_msg

    # Knee angle
    def _get_knee_angle(self, lm, width, height):
        sides = [(23, 25, 27), (24, 26, 28)]
        angles = []
        for hip_i, knee_i, ankle_i in sides:
            if not landmarks_visible(lm, [hip_i, knee_i, ankle_i], VISIBILITY_THRESHOLD):
                continue
            angles.append(calculate_angle(
                get_point(lm, hip_i,   width, height),
                get_point(lm, knee_i,  width, height),
                get_point(lm, ankle_i, width, height),
            ))
        return sum(angles) / len(angles) if angles else None

    def _sync_cue(self, user_angle, ref_angle):
        if ref_angle is None:
            return MSG_FOLLOW
        if user_angle > ref_angle + SYNC_ANGLE_TOLERANCE:
            return MSG_FOLLOW_DOWN
        if user_angle < ref_angle - SYNC_ANGLE_TOLERANCE:
            return MSG_FOLLOW_UP
        return MSG_FOLLOW

    # Form checks 
    def _check_form(self, lm, user_angle, ref_angle, width, height):
        ref_is_squatting = (ref_angle is not None and ref_angle < 130)
        if ref_is_squatting and user_angle > DEPTH_MIN_ANGLE:
            return MSG_GO_LOWER

        lean = self._check_torso_lean(lm)
        if lean:
            return lean

        if user_angle > 140:
            stance = self._check_stance_width(lm)
            if stance:
                return stance

        return None

    def _check_torso_lean(self, lm):
        if not landmarks_visible(lm, [11, 12, 23, 24], VISIBILITY_THRESHOLD):
            return None
        shoulder_x = (lm[11].x + lm[12].x) / 2.0
        hip_x      = (lm[23].x + lm[24].x) / 2.0
        if abs(shoulder_x - hip_x) > TORSO_LEAN_LIMIT:
            return MSG_BACK
        return None

    def _check_stance_width(self, lm):
        if not landmarks_visible(lm, [11, 12, 27, 28], VISIBILITY_THRESHOLD):
            return None
        shoulder_width = abs(lm[11].x - lm[12].x)
        foot_width     = abs(lm[27].x - lm[28].x)
        if shoulder_width < 1e-4:
            return None
        ratio = foot_width / shoulder_width
        if ratio < STANCE_MIN_RATIO:
            return MSG_WIDEN
        if ratio > STANCE_MAX_RATIO:
            return MSG_NARROW
        return None

    # Rep counting 
    def _detect_rep(self, user_angle):
        if user_angle is None:
            return
        if user_angle < SQUAT_DOWN_ANGLE and self.phase == "UP":
            self.phase = "DOWN"
        elif user_angle > SQUAT_UP_ANGLE and self.phase == "DOWN":
            self.phase = "UP"
            self.rep_count += 1