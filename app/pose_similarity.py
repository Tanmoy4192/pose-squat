import numpy as np
from collections import deque

# Landmarks used for similarity comparison.
# We intentionally exclude hands/feet (15,16,27,28) as they
# are often occluded and add noise to the comparison.
SIMILARITY_INDICES = [
    11, 12,   # shoulders
    13, 14,   # elbows
    23, 24,   # hips
    25, 26,   # knees
]

VISIBILITY_THRESHOLD = 0.5


def normalize_landmarks(landmarks):
    """
    Normalize landmarks to a canonical coordinate space:

    1. Center  — subtract the hip midpoint (more stable than shoulder center
                 because hips move less during a squat and are rarely occluded).
    2. Scale   — divide by the average of shoulder-width and hip-width.
                 Using two measurements makes the scale robust: if one pair
                 is partially occluded the other still anchors the scale.
    3. Filter  — zero out any landmark whose visibility < threshold so
                 invisible joints don't pollute the similarity score.

    Returns a 1-D float array of length len(SIMILARITY_INDICES) * 2.
    """
    pts = []
    weights = []

    for idx in SIMILARITY_INDICES:
        lm = landmarks[idx]
        vis = getattr(lm, "visibility", 1.0)
        pts.append([lm.x, lm.y])
        weights.append(1.0 if vis >= VISIBILITY_THRESHOLD else 0.0)

    pts = np.array(pts, dtype=float)        # shape (N, 2)
    weights = np.array(weights, dtype=float)

    # Hip midpoint as origin  (indices 4,5 in SIMILARITY_INDICES = lm 23,24)
    hip_idx_a = SIMILARITY_INDICES.index(23)
    hip_idx_b = SIMILARITY_INDICES.index(24)
    hip_mid = (pts[hip_idx_a] + pts[hip_idx_b]) / 2.0
    pts = pts - hip_mid

    # Scale: average of shoulder width and hip width
    sh_idx_a = SIMILARITY_INDICES.index(11)
    sh_idx_b = SIMILARITY_INDICES.index(12)
    shoulder_width = np.linalg.norm(pts[sh_idx_a] - pts[sh_idx_b])
    hip_width      = np.linalg.norm(pts[hip_idx_a] - pts[hip_idx_b])
    scale = (shoulder_width + hip_width) / 2.0
    scale = max(scale, 1e-6)               # avoid division by zero

    pts = pts / scale

    # Zero out invisible landmarks
    pts = pts * weights[:, np.newaxis]

    return pts.flatten()


def pose_similarity(user_lm, ref_lm):
    """
    Cosine similarity between the normalised user and reference poses.
    Returns a float in [-1, 1]; values above ~0.82 indicate a good match.
    """
    u = normalize_landmarks(user_lm)
    r = normalize_landmarks(ref_lm)

    mag = np.linalg.norm(u) * np.linalg.norm(r)
    if mag == 0:
        return 0.0

    # Clamp to [-1,1] to guard against floating-point drift
    return float(np.clip(np.dot(u, r) / mag, -1.0, 1.0))


class SmoothedSimilarity:
    """
    Wraps pose_similarity() with a rolling-average buffer to prevent
    flickering feedback caused by single noisy frames.

    Usage:
        smoother = SmoothedSimilarity(window=5)
        score = smoother.update(user_lm, ref_lm)   # returns smoothed float
    """

    def __init__(self, window: int = 5):
        self._buffer = deque(maxlen=window)

    def update(self, user_lm, ref_lm) -> float:
        raw = pose_similarity(user_lm, ref_lm)
        self._buffer.append(raw)
        return float(sum(self._buffer) / len(self._buffer))

    def reset(self):
        self._buffer.clear()