import cv2


class Camera:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def read_frame(self):
        return self.cap.read()

    def release(self) -> None:
        if self.cap:
            self.cap.release()