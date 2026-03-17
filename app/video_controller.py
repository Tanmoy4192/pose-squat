import cv2


class ReferenceVideo:

    def __init__(self, path):

        self.cap = cv2.VideoCapture(path)
        self.paused = False
        self.last_frame = None

    def read(self):

        if self.paused and self.last_frame is not None:
            return self.last_frame

        ret, frame = self.cap.read()

        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()

        self.last_frame = frame

        return frame

    def pause(self):

        self.paused = True

    def resume(self):

        self.paused = False