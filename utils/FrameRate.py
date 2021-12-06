import time

from src.config import FRAME_RATE


class FrameRate:
    def __init__(self, verbose=False):
        self.last_time = time.time()
        self.verbose = verbose

    def update(self):
        time_elapsed = time.time() - self.last_time
        while time_elapsed <= 1. / FRAME_RATE:
            time_elapsed = time.time() - self.last_time

        if self.verbose:
            print('Frame rate: %.4f' % (1 / time_elapsed))

        self.last_time = time.time()
