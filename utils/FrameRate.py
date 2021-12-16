import time

from src import DEFAULT_FRAME_RATE


class FrameRate:
    def __init__(self, verbose=False, frame_rate=DEFAULT_FRAME_RATE):
        self.last_time = time.time()
        self.verbose = verbose
        self.frame_rate = frame_rate

    def update(self):
        time_elapsed = time.time() - self.last_time
        while time_elapsed <= 1. / self.frame_rate:
            time_elapsed = time.time() - self.last_time

        if self.verbose:
            print('Frame rate: %.4f' % (1 / time_elapsed))

        self.last_time = time.time()
