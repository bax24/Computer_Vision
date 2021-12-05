import time

from src.config import FRAME_RATE


class FrameRate:
    def __init__(self):
        self.previous_frame = time.time()

    def update(self):
        time_elapsed = time.time() - self.previous_frame
        while time_elapsed <= 1. / FRAME_RATE:
            time_elapsed = time.time() - self.previous_frame
