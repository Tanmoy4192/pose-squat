import cv2
import time
import numpy as np
import mediapipe as mp

from camera import Camera
from pose_engine import PoseEngine
from video_controller import ReferenceVideo
from reference_analyzer import ReferenceAnalyzer
from squat_analyzer import WorkoutController
from ui_renderer import (
    draw_alert,
    draw_rep_counter,
    draw_start_overlay,
    draw_exercise_intro,
    draw_phase_indicator,
    draw_similarity_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_PATH     = "models/pose_landmarker_full.task"
REF_VIDEO_PATH = "reference/squat.mp4"

# ── Timing ─────────────────────────────────────────────────────────────────────

INTRO_DURATION     = 8   # seconds — exercise intro card
COUNTDOWN_DURATION = 3   # seconds — "Get Ready" countdown

# ── Font constants for angle overlay ──────────────────────────────────────────

FONT       = cv2.FONT_HERSHEY_SIMPLEX
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0,   0,   0)


def build_mp_image(frame):
    """Convert a BGR cv2 frame to a MediaPipe SRGB Image."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def letterbox(frame, target_w, target_h):
    """
    Resize frame preserving aspect ratio, padding with black bars.
    Prevents landmark coordinate distortion caused by stretching.
    """
    src_h, src_w = frame.shape[:2]
    scale   = min(target_w / src_w, target_h / src_h)
    new_w   = int(src_w * scale)
    new_h   = int(src_h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas  = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off   = (target_w - new_w) // 2
    y_off   = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def draw_angle_overlay(frame, user_angle, ref_angle):
    """
    Shows live knee angles for both user and reference in top-left of
    each panel. Helps the user understand exactly where they are vs
    where the reference is.
    """
    h, w = frame.shape[:2]
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (310, 60), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    u_str = f"You:  {user_angle:5.1f}" if user_angle is not None else "You:  --"
    r_str = f"Ref:  {ref_angle:5.1f}"  if ref_angle  is not None else "Ref:  --"

    cv2.putText(frame, u_str, (18, 32),  FONT, 0.65, COLOR_BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, u_str, (18, 32),  FONT, 0.65, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, r_str, (165, 32), FONT, 0.65, COLOR_BLACK, 4, cv2.LINE_AA)
    cv2.putText(frame, r_str, (165, 32), FONT, 0.65, (100, 220, 255), 2, cv2.LINE_AA)


def main():
    # ── Initialise components ─────────────────────────────────────────────────
    camera             = Camera(0)
    user_detector      = PoseEngine(MODEL_PATH)
    reference_video    = ReferenceVideo(REF_VIDEO_PATH)
    reference_analyzer = ReferenceAnalyzer(MODEL_PATH)
    controller         = WorkoutController()

    start_time      = time.time()
    last_similarity = 0.0
    last_user_angle = None
    last_ref_angle  = None

    while True:
        # ── Read camera ───────────────────────────────────────────────────────
        frame = camera.read()
        height, width, _ = frame.shape
        elapsed = time.time() - start_time

        # ── Intro card ────────────────────────────────────────────────────────
        if elapsed < INTRO_DURATION:
            draw_exercise_intro(frame)
            cv2.imshow("AI Pose Trainer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ── Countdown ─────────────────────────────────────────────────────────
        if elapsed < INTRO_DURATION + COUNTDOWN_DURATION:
            seconds = int(INTRO_DURATION + COUNTDOWN_DURATION - elapsed) + 1
            draw_start_overlay(frame, seconds)
            cv2.imshow("AI Pose Trainer", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # ── Live session ──────────────────────────────────────────────────────

        # 1. User pose detection (async)
        mp_image  = build_mp_image(frame)
        timestamp = int(time.time() * 1000)
        user_detector.detect_async(mp_image, timestamp)

        # 2. Read thread-safe result
        user_result    = user_detector.get_latest_result()
        user_landmarks = None
        if user_result and user_result.pose_landmarks:
            user_landmarks = user_result.pose_landmarks[0]

        # 3. Reference frame (aspect-ratio-safe resize)
        ref_frame     = reference_video.read()
        ref_frame     = letterbox(ref_frame, width, height)
        ref_landmarks = reference_analyzer.extract(ref_frame)

        # 4. Evaluate — returns (advance_video, message, correct)
        advance_video, message, correct = controller.evaluate(
            user_landmarks,
            ref_landmarks,
            width,
            height,
        )

        # 5. Cache angles for HUD (read from controller's last computation)
        last_user_angle = controller._get_knee_angle(user_landmarks, width, height) \
                          if user_landmarks else None
        last_ref_angle  = controller._get_knee_angle(ref_landmarks,  width, height) \
                          if ref_landmarks  else None

        # 6. Cache smoothed similarity for HUD
        buf = controller._smoother._buffer
        if buf:
            last_similarity = sum(buf) / len(buf)

        # 7. Advance or pause reference video
        if advance_video:
            reference_video.resume()
        else:
            reference_video.pause()

        # 8. Draw skeleton on user frame (white = correct, red = incorrect)
        frame = user_detector.draw_skeleton(frame, correct)

        # 9. User panel HUD
        draw_phase_indicator(frame, controller.phase)
        draw_similarity_score(frame, last_similarity)
        draw_angle_overlay(frame, last_user_angle, last_ref_angle)

        # 10. Reference panel HUD
        draw_alert(ref_frame, message)
        draw_rep_counter(ref_frame, controller.rep_count)

        # 11. Combine and display
        combined = cv2.hconcat([frame, ref_frame])
        cv2.imshow("AI Pose Trainer", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()