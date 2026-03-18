import cv2
class Camera:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera not accessible")
        frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        self.cap.release()