import time

import cv2

from buffers.VideoReader import VideoReader
from src.buffers.ModelComparatorBuffer import ModelComparatorBuffer
from src.models.ExponentialReferenceFrame import ExponentialReferenceFrame
from src.models.GaussianDensityEstimation import GaussianDensityEstimation
from src.models.GaussianMixtureModel import GaussianMixtureModel
from utils.FrameRate import FrameRate
from utils.utils import get_videos_path


def init_background_subtractors(video_list):
    video_path = video_list[0]
    cap = cv2.VideoCapture(video_path)

    gde = GaussianDensityEstimation(
        memory_size=20,
        frame_dim=FRAME_DIM,
        frame_scale=0.25
    )

    gmm = GaussianMixtureModel(
        memory_size=20,
        frame_dim=FRAME_DIM,
        frame_scale=0.25
    )

    gde.fill_memory(cap)
    gmm.fill_memory(cap)

    erf = ExponentialReferenceFrame(
        video_path=video_path,
        gaussian_kernel=(5, 5),
        threshold=25,
        alpha=0.05
    )

    models = [gde, gmm, erf]
    model_names = ["Gaussian Density Estimator",
                   "Gaussian Mixture Model",
                   "Exponential Reference Frame Method"]

    return models, model_names


def main(video_list):
    # Build all buffers
    buffer_video = VideoReader(
        path_list=video_list,
        buff_size=BUFFER_SIZE
    )

    back_subtractors, back_names = init_background_subtractors(video_list)
    buffer_model_comparator = ModelComparatorBuffer(
        buffer_video,
        back_subtractors,
        BUFFER_SIZE
    )

    buffer_video.start()
    buffer_model_comparator.start()

    frame_rate = FrameRate(verbose=False, frame_rate=FRAME_RATE)

    while buffer_model_comparator.more():

        frame_rate.update()
        if not buffer_model_comparator.buffer.empty():
            frame, masks, computation_time, frame_idx = buffer_model_comparator.read()

            cv2.imshow("Frame", frame)

            print("\n-----------------------")
            for mask, model_name, elapsed_time in zip(masks, back_names, computation_time):
                print("{}: {}".format(model_name, elapsed_time))
                cv2.imshow(model_name, mask)

            key = cv2.waitKey(2)
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(-1)

        else:
            time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    BUFFER_SIZE = 500
    FRAME_DIM = (240, 1600)
    FRAME_RATE = 15

    main(get_videos_path())
