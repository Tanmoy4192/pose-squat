import cv2

# ── Typography & layout constants ─────────────────────────────────────────────

FONT       = cv2.FONT_HERSHEY_SIMPLEX
COLOR_WHITE   = (255, 255, 255)
COLOR_CYAN    = (255, 255,   0)   # BGR → yellow-green; visible on dark & light bg
COLOR_GREEN   = ( 50, 205,  50)
COLOR_RED     = (  0,   0, 255)
COLOR_ORANGE  = (  0, 165, 255)
COLOR_BLACK   = (  0,   0,   0)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _semi_rect(frame, x1, y1, x2, y2, alpha=0.55):
    """Draw a semi-transparent black rectangle (background for text)."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _put(frame, text, x, y, scale=1.0, color=COLOR_WHITE, thickness=2):
    """Single-line text with a thin black stroke for legibility on any background."""
    cv2.putText(frame, text, (x, y), FONT, scale, COLOR_BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, scale, color,      thickness,     cv2.LINE_AA)


# ── Public drawing functions ──────────────────────────────────────────────────

def draw_alert(frame, text):
    """
    Feedback banner at the bottom of the reference panel.
    Background tint changes colour based on the message severity:
      - "Good form"  → green tint
      - anything else → red tint
    """
    h, w, _ = frame.shape
    box_y1, box_y2 = h - 120, h - 40

    good = text.lower().startswith("good")
    tint_color = (0, 60, 0) if good else (0, 0, 60)

    overlay = frame.copy()
    cv2.rectangle(overlay, (30, box_y1), (w - 30, box_y2), tint_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    text_color = COLOR_GREEN if good else COLOR_RED
    text_size  = cv2.getTextSize(text, FONT, 1.0, 2)[0]
    tx = (w - text_size[0]) // 2          # centre-align text
    _put(frame, text, tx, h - 70, scale=1.0, color=text_color)


def draw_rep_counter(frame, reps):
    """Rep counter in the top-left of the reference panel."""
    _semi_rect(frame, 20, 20, 180, 80)
    _put(frame, f"Reps: {reps}", 36, 66, scale=1.4, color=COLOR_CYAN, thickness=3)


def draw_phase_indicator(frame, phase):
    """
    Shows the current squat phase (UP / DOWN) in the top-right of the
    user panel.  Helps with debugging rep-counting state.
    """
    h, w, _ = frame.shape
    color  = COLOR_ORANGE if phase == "DOWN" else COLOR_GREEN
    label  = f"Phase: {phase}"
    tw     = cv2.getTextSize(label, FONT, 0.8, 2)[0][0]
    _semi_rect(frame, w - tw - 50, 20, w - 20, 68)
    _put(frame, label, w - tw - 36, 56, scale=0.8, color=color)

def draw_similarity_score(frame, score):
    h, w, _ = frame.shape
    if score >= 0.82:
        color = COLOR_GREEN
    elif score >= 0.70:
        color = COLOR_ORANGE
    else:
        color = COLOR_RED

    label = f"Sim: {score:.2f}"
    _semi_rect(frame, 20, h - 70, 200, h - 20)
    _put(frame, label, 36, h - 34, scale=0.85, color=color)

def draw_start_overlay(frame, seconds):
    """Full-screen countdown overlay shown before the session begins."""
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    _put(frame, "Get Ready",
         w // 2 - 120, h // 2 - 40,
         scale=1.5, color=COLOR_WHITE, thickness=3)

    _put(frame, str(seconds),
         w // 2 - 28, h // 2 + 70,
         scale=3.0, color=COLOR_CYAN, thickness=5)


def draw_exercise_intro(frame):
    """
    Intro card shown for the first 8 seconds.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), COLOR_BLACK, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lines = [
        ("Exercise 1", 1.0, COLOR_CYAN),
        ("Bodyweight Squat", 1.2, COLOR_WHITE),
        ("", 0.8, COLOR_WHITE),
        ("Stand feet shoulder-width apart", 0.75, COLOR_WHITE),
        ("Keep your chest up & back straight", 0.75, COLOR_WHITE),
        ("Lower hips until knees reach 90 deg", 0.75, COLOR_WHITE),
        ("Drive through heels to stand", 0.75, COLOR_WHITE),
    ]

    y = h // 2 - 130
    for text, scale, color in lines:
        if text == "":
            y += 24
            continue
        tw = cv2.getTextSize(text, FONT, scale, 2)[0][0]
        x  = (w - tw) // 2
        _put(frame, text, x, y, scale=scale, color=color)
        y += int(scale * 52)