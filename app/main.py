import cv2
import time
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
    draw_exercise_intro
)


def main():

    camera = Camera(0)

    user_pose_detector = PoseEngine("models/pose_landmarker_full.task")
    reference_pose_detector = PoseEngine("models/pose_landmarker_full.task")

    reference_video = ReferenceVideo("reference/squat.mp4")

    reference_analyzer = ReferenceAnalyzer(reference_pose_detector)

    controller = WorkoutController()

    start_time = time.time()

    intro_duration = 8
    countdown_duration = 3

    while True:

        frame = camera.read()
        height, width, _ = frame.shape
        elapsed = time.time() - start_time

        if elapsed < intro_duration:

            draw_exercise_intro(frame)

            cv2.imshow("AI Pose Trainer", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        if elapsed < intro_duration + countdown_duration:

            seconds = int(intro_duration + countdown_duration - elapsed) + 1

            draw_start_overlay(frame, seconds)

            cv2.imshow("AI Pose Trainer", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = int(time.time() * 1000)

        user_pose_detector.detect_async(mp_image, timestamp)

        user_landmarks = None

        if user_pose_detector.latest_result and user_pose_detector.latest_result.pose_landmarks:

            user_landmarks = user_pose_detector.latest_result.pose_landmarks[0]

        ref_frame = reference_video.read()

        ref_frame = cv2.resize(ref_frame, (width, height))

        ref_landmarks = reference_analyzer.extract(ref_frame)

        correct, message = controller.evaluate(
            user_landmarks,
            ref_landmarks,
            width,
            height
        )

        frame = user_pose_detector.draw_skeleton(frame, correct)

        if correct:
            reference_video.resume()
        else:
            reference_video.pause()

        draw_alert(ref_frame, message)
        draw_rep_counter(ref_frame, controller.rep_count)

        combined = cv2.hconcat([frame, ref_frame])

        cv2.imshow("AI Pose Trainer", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()