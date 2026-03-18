from pose_similarity import SmoothedSimilarity
from utils import calculate_angle, landmarks_visible, get_point

#  Thresholds 

# Rep state machine
SQUAT_DOWN_ANGLE = 100   # knee angle below this -> phase DOWN
SQUAT_UP_ANGLE = 155   # knee angle above this -> phase UP (standing)

# Form checks
DEPTH_MIN_ANGLE = 120   # if user knee > this while ref is squatting → "Go lower"
TORSO_LEAN_LIMIT = 0.15  # normalised |shoulder_x_mid - hip_x_mid|
STANCE_MIN_RATIO = 0.70  # foot_width / shoulder_width minimum
STANCE_MAX_RATIO = 1.40  # foot_width / shoulder_width maximum

# How close user knee angle must be to ref knee angle to reference video
SYNC_ANGLE_TOLERANCE = 25

VISIBILITY_THRESHOLD = 0.5

class WorkoutController:
    def __init__(self):
        self.rep_count = 0
        self.phase = "UP"
        self._smoother = SmoothedSimilarity(window=5)

    # Public entry point 

    def evaluate(self, user_lm, ref_lm, width, height):
        """
        logic:
          - Compute user and reference knee angles independently
          - Reference video advances ONLY when user angle is within
            SYNC_ANGLE_TOLERANCE of the reference angle
          - Form checks use reference angle context so "Go lower" only
            fires when the reference is actually squatting
          - Rep counted by user's own angle state machine
        """
        if user_lm is None or ref_lm is None:
            return False, "Detecting pose...", False

        user_angle = self._get_knee_angle(user_lm, width, height)
        ref_angle  = self._get_knee_angle(ref_lm,  width, height)

        if user_angle is None:
            return False, "Detecting pose...", False

        if ref_angle is not None:
            in_sync = abs(user_angle - ref_angle) <= SYNC_ANGLE_TOLERANCE
        else:
            in_sync = True   # if can't measure ref then don't block the video

        # Form checks 
        form_msg = self._check_form(user_lm, user_angle, ref_angle, width, height)
        correct  = (form_msg is None)

        # feedback message
        if not in_sync:
            # Sync cue takes priority — guide user to match reference position
            message = self._sync_cue(user_angle, ref_angle)
            correct = False
        elif form_msg:
            message = form_msg
        else:
            message = "Good form"

        # Rep counting (always runs) 
        self._detect_rep(user_angle)

        # Video advances only when user is both in sync and has correct form
        advance_video = in_sync and correct
        return advance_video, message, correct

    # Knee angle 

    def _get_knee_angle(self, lm, width, height):
        """Average knee angle across both visible legs. Returns None if neither visible."""
        sides  = [(23, 25, 27), (24, 26, 28)]
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
        """Direction cue to bring user back in sync with reference."""
        if ref_angle is None:
            return "Follow the mentor pose"
        if user_angle > ref_angle + SYNC_ANGLE_TOLERANCE:
            return "Go lower follow the mentor"
        if user_angle < ref_angle - SYNC_ANGLE_TOLERANCE:
            return "Come up follow the mentor"
        return "Follow the mentor pose"

    # Form checks 

    def _check_form(self, lm, user_angle, ref_angle, width, height):
        # Depth — only meaningful when reference is itself in squat position
        ref_is_squatting = (ref_angle is not None and ref_angle < 130)
        if ref_is_squatting and user_angle > DEPTH_MIN_ANGLE:
            return "Go lower into squat"

        # Torso lean — always check
        lean = self._check_torso_lean(lm)
        if lean:
            return lean

        # Stance width — only when standing upright 
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
            return "Keep your back straight"
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
            return "Widen your stance"
        if ratio > STANCE_MAX_RATIO:
            return "Bring feet closer together"
        return None

    # Rep counting 

    def _detect_rep(self, user_angle):
        """
        DOWN at < 100°, UP at > 155° — large gap prevents noise double-counting.
        """
        if user_angle is None:
            return
        if user_angle < SQUAT_DOWN_ANGLE and self.phase == "UP":
            self.phase = "DOWN"
        elif user_angle > SQUAT_UP_ANGLE and self.phase == "DOWN":
            self.phase = "UP"
            self.rep_count += 1