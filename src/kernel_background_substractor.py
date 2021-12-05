import numpy as np


class KernelBGSubstractor:
    def __init__(self, memory_size, image_shape):
        self.frame_counter = 0
        self.memory_size = memory_size
        self.memory = np.zeros([image_shape[0], image_shape[1], memory_size], np.uint8)

    def __call__(self, frame):
        assert frame.shape == (self.memory.shape[0], self.memory.shape[1]), "Frame of incorrect shape! \n"

        idx = self.frame_counter % self.memory_size
        self.memory[:, :, idx] = frame

        std = np.std(self.memory, axis=2)
        weight = 1. / self.frame_counter if self.frame_counter < self.memory_size else 1. / self.memory_size
        print(std.shape)


if __name__ == "__main__":
    substractor = KernelBGSubstractor(50, (50, 50))
    substractor(np.zeros((50, 50), np.uint8))
