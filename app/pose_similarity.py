import numpy as np


def normalize_landmarks(landmarks):

    pts = []

    for lm in landmarks:
        pts.append([lm.x, lm.y])

    pts = np.array(pts)

    center = pts[11]

    pts = pts - center

    scale = np.linalg.norm(pts[12] - pts[11])

    if scale == 0:
        scale = 1

    pts = pts / scale

    return pts.flatten()


def pose_similarity(user_lm, ref_lm):

    u = normalize_landmarks(user_lm)
    r = normalize_landmarks(ref_lm)

    dot = np.dot(u, r)
    mag = np.linalg.norm(u) * np.linalg.norm(r)

    if mag == 0:
        return 0

    return dot / mag