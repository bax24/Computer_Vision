from scipy.stats import norm

from BackgroundSubstractor import *


# https://www.cs.toronto.edu/~duvenaud/cookbook/
# Interesting kernels

class GaussianDensityEstimation(BackgroundSubstractor):
    def __init__(self, memory_size, frame, frame_scale=1., num_workers=1, num_chunks=None):
        super().__init__(memory_size, frame, frame_scale, num_workers, num_chunks)

    def __call__(self, frame):
        frame = resize(frame, self.frame_scale)
        dim = frame.shape
        frame = np.ravel(frame)

        distributed_memory = np.array_split(self.previous_frames, self.num_chunks, axis=1)
        distributed_frame = np.array_split(frame, self.num_chunks, axis=0)

        distributed_data = [t for t in zip(distributed_frame, distributed_memory)]
        workers_result = self.worker_pool.starmap(get_pdf, distributed_data)
        results = np.hstack(workers_result)

        return resize(results.reshape(dim), 1. / self.frame_scale)

    def fill_memory(self, cap: cv2.VideoCapture):
        super(GaussianDensityEstimation, self).fill_memory(cap)


def get_pdf(frame, memory):
    stack_size = memory.shape[0]
    frame = np.array([frame] * stack_size, dtype=np.uint8)

    std = [np.std(memory, axis=0, dtype=np.float16)] * stack_size
    return norm.pdf(frame, loc=memory, scale=std).mean(axis=0)
