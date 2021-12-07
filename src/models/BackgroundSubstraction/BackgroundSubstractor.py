from multiprocessing import Pool

import cv2
import numpy as np


class BackgroundSubstractor:
    def __init__(self, memory_size, frame, frame_scale=1., num_workers=1, num_chunks=None):
        self.frame_counter = 0
        self.memory_size = memory_size

        reshaped_frame = resize(frame, frame_scale)
        self.previous_frames = np.zeros([self.memory_size, reshaped_frame.size], np.uint8)
        self.frame_scale = frame_scale

        self.num_worker = num_workers
        self.worker_pool = Pool(num_workers)
        self.num_chunks = num_workers if num_chunks is None else num_chunks

    def __del__(self):
        self.worker_pool.close()

    def fill_memory(self, cap: cv2.VideoCapture):
        cap.set(2, 0.0)  # Rest the head to the start of the video
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = resize(frame, self.frame_scale)

        while ret is not False and self.frame_counter < self.memory_size:
            self.push(frame.ravel())
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = resize(frame, self.frame_scale)

    def push(self, frame):
        idx = self.frame_counter % self.memory_size
        self.previous_frames[idx, :] = frame
        self.frame_counter += 1


def resize(frame, factor):
    width = int(frame.shape[1] * factor)
    height = int(frame.shape[0] * factor)
    return cv2.resize(frame, (width, height))


def remove_noise(frame, kernel, iterations):
    frame = cv2.erode(frame, kernel, iterations)
    frame = cv2.dilate(frame, kernel, iterations)
    return frame
