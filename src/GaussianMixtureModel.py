from multiprocessing import Pool

import cv2
import numpy as np
from scipy.stats import norm

from config import DATASET_ROOT
from utils.FrameRate import FrameRate
from utils.utils import open_or_exit


class GaussianMixtureModel:
    def __init__(self, memory_size, frame, num_workers=1, num_chunks=None):
        self.frame_counter = 0
        self.memory_size = memory_size
        self.previous_frames = np.zeros([self.memory_size, frame.size], np.uint8)

        self.num_worker = num_workers
        self.worker_pool = Pool(num_workers)
        self.num_chunks = num_workers if num_chunks is None else num_chunks

        self.push(frame.ravel())

        print("Number of elements: ", self.previous_frames.size)

    def __call__(self, frame):
        dim = frame.shape
        frame = np.ravel(frame)
        observed_frames = min(self.memory_size, self.frame_counter)

        results = self.map_reduce(frame, self.previous_frames[:observed_frames, :])
        self.push(frame)
        return results.reshape(dim)

    def __del__(self):
        self.worker_pool.close()

    def map_reduce(self, frame, previous_frames):
        distributed_memory = np.array_split(previous_frames, self.num_chunks, axis=1)
        distributed_frame = np.array_split(frame, self.num_chunks, axis=0)

        distributed_data = [t for t in zip(distributed_frame, distributed_memory)]
        workers_result = self.worker_pool.starmap(get_pdf, distributed_data)
        return np.hstack(workers_result)

    def push(self, frame):
        idx = self.frame_counter % self.memory_size
        self.previous_frames[idx, :] = frame
        self.frame_counter += 1


def get_pdf(frame, memory):
    mean = np.mean(memory, axis=0, dtype=np.uint8)
    std = np.std(memory, axis=0, dtype=np.float16)
    return norm.pdf(frame, loc=mean, scale=std)

    # def __call__(self, frame):
    #     idx = min(self.memory_size, self.frame_counter)
    #
    #     mean = self.memory[:idx, :, :].mean(axis=0)
    #     std = self.memory[:idx, :, :].std(axis=0)
    #     prob = norm.pdf(frame, loc=mean, scale=std)
    #
    #     self.push(frame)
    #
    #     return prob


def resize(frame, factor):
    width = int(frame.shape[1] * factor)
    height = int(frame.shape[0] * factor)
    return cv2.resize(frame, (width, height))


if __name__ == "__main__":
    path = f"{DATASET_ROOT}/raw2/images/CV2021_GROUP05/group5.mp4"
    print(path)
    cap = open_or_exit(path)

    _, frame_ = cap.read()
    frame_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)

    scale_ = 0.5
    thresh = 1E-15
    memory_size = 5
    num_workers_ = 6
    substractor = GaussianMixtureModel(memory_size, resize(frame_, scale_), num_workers=num_workers_)

    frame_rate = FrameRate(verbose=True, frame_rate=15)
    frame_counter = 0
    while cap.isOpened():

        ret, frame_ = cap.read()
        frame_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)

        if ret:
            frame_ = resize(frame_, scale_)
            # substractor(frame_)
            prob_ = substractor(frame_)

            mask = (prob_ >= thresh).astype(np.uint8)
            mask *= 255
            frame_ = resize(frame_, 1 / scale_)
            prob_ = resize(mask, 1 / scale_)

            frame_rate.update()
            cv2.imshow("Frame", frame_)
            cv2.imshow("Mask", prob_)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
