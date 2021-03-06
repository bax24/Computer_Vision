from scipy.stats import norm

from src.models.BackgroundSubtractor import *


class GaussianMixtureModel(BackgroundSubtractor):
    def __init__(self,
                 memory_size,
                 frame_dim,
                 frame_scale=1.,
                 num_workers=4,
                 num_chunks=20,
                 prob_threshold=1e-25,
                 clean_params=((20, 20), 5)):

        super().__init__(memory_size,
                         frame_dim,
                         frame_scale,
                         num_workers,
                         num_chunks,
                         clean_params)

        self.prob_threshold = prob_threshold
        self.mean = None
        self.std = None

    def fill_memory(self, cap: cv2.VideoCapture):
        super(GaussianMixtureModel, self).fill_memory(cap)
        self.mean = self.previous_frames.mean(axis=0, dtype=np.float16)
        self.std = self.previous_frames.std(axis=0, dtype=np.float16)

    def __call__(self, frame):
        frame = resize(frame, self.frame_scale)
        dim = frame.shape

        frame = np.ravel(frame)

        distributed_frame = np.array_split(frame, self.num_chunks, axis=0)
        distributed_mean = np.array_split(self.mean, self.num_chunks, axis=0)
        distributed_std = np.array_split(self.std, self.num_chunks, axis=0)

        distributed_data = [t for t in zip(distributed_frame, distributed_mean, distributed_std)]
        workers_result = self.worker_pool.starmap(get_pdf, distributed_data)

        result = np.hstack(workers_result)
        result = resize(result.reshape(dim), 1. / self.frame_scale)
        result = (result <= self.prob_threshold).astype(np.uint8) * 255
        return self.clean_frame(result)


def get_pdf(frame, mean, std):
    return norm.pdf(frame, loc=mean, scale=std)
