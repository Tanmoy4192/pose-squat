import cv2
import mediapipe as mp

from pose_engine import ImagePoseEngine


class ReferenceAnalyzer:
    """
    Extracts pose landmarks from reference video frames using the synchronous ImagePoseEngine (RunningMode.IMAGE).
    """
    def __init__(self, model_path: str):
        # Guard against accidentally passing a PoseEngine object instead
        # of a path string (a common mistake when refactoring from the old API).
        if not isinstance(model_path, str):
            raise TypeError(
                f"ReferenceAnalyzer expects a model file path (str), "
                f"but received {type(model_path).__name__}. "
                f"Pass the path string directly, e.g. "
                f"ReferenceAnalyzer('models/pose_landmarker_full.task')"
            )
        self._engine = ImagePoseEngine(model_path)
        self._last_landmarks = None          # cache of last successful detection

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        landmarks = self._engine.detect(mp_image)

        if landmarks is not None:
            self._last_landmarks = landmarks   # update cache on success
        # else: keep _last_landmarks as-is 
        
        return self._last_landmarks