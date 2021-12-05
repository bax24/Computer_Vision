import numpy as np
import cv2

from config import DATASET_ROOT
from utils.FrameRate import FrameRate
from utils.utils import open_or_exit


def normalize_pdf(x, mean, sigma):
    return np.exp(-0.5 * (((x - mean) / sigma) ** 2)) / (np.sqrt(2 * np.pi) * sigma)


class GaussianMixtureModel:
    def __init__(self, frame):
        height, width = frame.shape

        self.mean = np.zeros([3, height, width], np.float64)
        self.mean[1, :, :] = frame

        self.variance = self.mean.copy()
        self.variance[:, :, :] = 400

        self.omega = self.mean.copy()
        self.omega[0, :, :] = 0
        self.omega[1, :, :] = 0
        self.omega[2, :, :] = 1

        self.omega_by_sigma = self.mean.copy()

        self.foreground = np.zeros([height, width], np.uint8)
        self.background = np.zeros([height, width], np.uint8)

        self.alpha = 0.3
        self.T = 0.5

    def __call__(self, cap):
        frame_rate = FrameRate()
        while cap.isOpened():
            frame_rate.update()

            ret, frame = cap.read()
            if ret:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = gray_frame.astype(np.float64)
                gray_frame = np.array([gray_frame] * 3)

                self.reset_variances()
                sigma = np.sqrt(self.variance)
                fitting_threshold = 2.5 * sigma

                cmp = cv2.absdiff(gray_frame, self.mean)
                gaussian_fit = np.where(cmp <= fitting_threshold)
                gaussian_nofit = np.where(cmp > fitting_threshold)


            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def reset_variances(self):
        self.variance[0][np.where(self.variance[0] < 1)] = 10
        self.variance[1][np.where(self.variance[1] < 1)] = 5
        self.variance[2][np.where(self.variance[2] < 1)] = 1


if __name__ == "__main__":
    path = f"{DATASET_ROOT}/raw2/images/CV2021_GROUP05/group5.mp4"
    _cap = open_or_exit(path)

    _, _frame = _cap.read()
    _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
    print(_frame.shape)
    gmm = GaussianMixtureModel(_frame)
    gmm(_cap)

# TODO: evaluate this technique
