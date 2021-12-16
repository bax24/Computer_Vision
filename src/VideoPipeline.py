import time

import cv2

from buffers.BackgroundSubtractorBuffer import BackgroundSubtractorBuffer
from buffers.DropletExtractorBuffer import DropletExtractorBuffer
from buffers.VideoReader import VideoReader
from models.ExponentialReferenceFrame import ExponentialReferenceFrame
from utils.FrameRate import FrameRate
from utils.utils import get_videos_path


def init_background_subtractor(video_list):
    ERF = ExponentialReferenceFrame(
        video_path=video_list[0],
        gaussian_kernel=(5, 5),
        threshold=25,
        alpha=0.05
    )
    return ERF


def main(video_list):
    # Build all buffers
    buffer_video = VideoReader(
        path_list=video_list,
        buff_size=BUFFER_SIZE
    )

    buffer_background = BackgroundSubtractorBuffer(
        input_buffer=buffer_video,
        background_subtractor=init_background_subtractor(video_list),
        buff_size=BUFFER_SIZE
    )

    buffer_droplet = DropletExtractorBuffer(
        input_buffer=buffer_background,
        buff_size=BUFFER_SIZE,
        droplet_threshold=30,
        droplet_min_width=250
    )

    buffer_video.start()
    buffer_background.start()
    buffer_droplet.start()

    frame_rate = FrameRate(verbose=True, frame_rate=FRAME_RATE)

    while buffer_droplet.more():

        frame_rate.update()
        if not buffer_droplet.buffer.empty():

            frame, mask, droplet, frame_idx = buffer_droplet.read()
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)

            if not (droplet is None):
                cv2.imshow("Droplet", droplet)

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
    FRAME_RATE = 300

    main(get_videos_path())
