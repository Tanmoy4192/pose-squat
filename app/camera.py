import cv2

class Camera:

    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

    def read(self):

        success, frame = self.cap.read()

        if not success:
            return None

        return frame