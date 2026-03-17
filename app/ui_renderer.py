import cv2


def draw_alert(frame, text):

    overlay = frame.copy()
    h, w, _ = frame.shape

    cv2.rectangle(
        overlay,
        (30, h - 120),
        (w - 30, h - 40),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame,
        text,
        (60, h - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )


def draw_rep_counter(frame, reps):

    cv2.putText(
        frame,
        f"Reps: {reps}",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )


def draw_start_overlay(frame, seconds):

    overlay = frame.copy()
    h, w, _ = frame.shape

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, h),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame,
        "Get Ready",
        (w // 2 - 120, h // 2 - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (255, 255, 255),
        3
    )

    cv2.putText(
        frame,
        str(seconds),
        (w // 2 - 20, h // 2 + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (0, 255, 255),
        5
    )


def draw_exercise_intro(frame):

    overlay = frame.copy()
    h, w, _ = frame.shape

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, h),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lines = [
        "Exercise 1",
        "Bodyweight Squat",
        "",
        "Stand with feet shoulder width",
        "Keep your back straight",
        "Lower your hips down",
        "Bend knees then stand up"
    ]

    y_start = h // 2 - 120
    gap = 50

    for i, text in enumerate(lines):

        text_size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            2
        )[0]

        x = (w - text_size[0]) // 2
        y = y_start + i * gap

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )