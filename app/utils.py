import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b, formed by a -> b -> c.
    Returns degrees in range [0, 180].
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)   # prevent arccos domain errors
    return np.degrees(np.arccos(cosine))


def landmarks_visible(landmarks, indices, threshold=0.5):
    for idx in indices:
        lm = landmarks[idx]
        vis = getattr(lm, "visibility", None)
        if vis is not None and vis < threshold:
            return False
    return True

def get_point(lm, idx, width, height):
    """
    Convert a normalised MediaPipe landmark to pixel coordinates.
    Returns (x_px, y_px) as a tuple of floats.
    """
    return (lm[idx].x * width, lm[idx].y * height)