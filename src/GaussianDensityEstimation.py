import cv2
import numpy as np
from scipy.stats import norm

from config import DATASET_ROOT
from utils.FrameRate import FrameRate
from utils.utils import open_or_exit


class GaussianDensityEstimation:
    def __init__(self, memory_size, frame):
        self.frame_counter = 0
        self.memory_size = memory_size
        self.memory = np.zeros([self.memory_size, frame.shape[0], frame.shape[1]], np.uint8)

        self.push(frame)

        print("Number of elements: ", self.memory.size)

    def __call__(self, frame):
        idx = min(self.memory_size, self.frame_counter)
        stacked_frame = np.array([frame] * idx)

        mean = self.memory[:idx, :, :]
        std = [np.std(mean, axis=0, dtype=np.float16)] * idx
        prob = norm.pdf(stacked_frame, loc=mean, scale=std).mean(axis=0)

        self.push(frame)

        return prob

    def push(self, frame):
        idx = self.frame_counter % self.memory_size
        self.memory[idx, :, :] = frame
        self.frame_counter += 1


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

    scale_ = 0.18
    thresh = 1E-10
    memory_size = 50
    substractor = GaussianDensityEstimation(memory_size, resize(frame_, scale_))

    frame_rate = FrameRate(verbose=False)
    while cap.isOpened():

        ret, frame_ = cap.read()
        frame_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)

        if ret:
            frame_ = resize(frame_, scale_)
            prob_ = substractor(frame_)

            mask = (prob_ <= thresh).astype(np.uint8)
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
