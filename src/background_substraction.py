import os
import time
from collections import deque

import cv2
import numpy as np

from src.config import DATASET_ROOT


#################################
# Defining filters
#################################
from utils.utils import open_or_exit


def average_filter(kx, ky, img):
    kernel = np.ones((ky, kx)) / (kx * ky)
    avg_image = cv2.filter2D(img, -1, kernel)

    return avg_image


def gaussian_filter(kx, ky, img):
    kernel = (ky, kx)
    gauss_image = cv2.GaussianBlur(img, kernel, 0)

    return gauss_image


#################################
# Background subtraction methods
#################################
def display_video(path):
    cap = open_or_exit(path)

    frame_rate = FrameRate()
    while cap.isOpened():
        frame_rate.update()

        prev = time.time()
        ret, frame = cap.read()
        if ret:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def compare_methods(video_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error while opening file, aborting... \n")
        exit(-1)

    # Read first frame as background
    ret, frame = cap.read()
    if not ret:
        print("Impossible to read video stream, aborting...\n")
        exit(-1)

    background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).copy()
    exponential_background = background.copy()
    last_frames = deque()
    last_frames.append(background)

    queue_size = 0
    frame_rate = FrameRate()
    while cap.isOpened():
        frame_rate.update()

        # Main process
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gaussian_filtered_gray = gaussian_filter(5, 5, frame)

            # Single frame reference method
            # mask0 = np.abs(frame - background) > threshold

            # Exponential update of reference frame method
            mask1 = np.abs(gaussian_filtered_gray - exponential_background) > threshold
            exponential_background = alpha * gaussian_filtered_gray + (1 - alpha) * exponential_background

            # Median of last frames
            # last_frames_median = np.median([frame_ for frame_ in last_frames], axis=0)
            # mask2 = np.abs(gaussian_filtered_gray - last_frames_median)
            # if queue_size >= last_frames_number:
            #     last_frames.popleft()
            # last_frames.append(gaussian_filtered_gray)

            # Displaying original sequence, mask and masked sequence
            cv2.imshow('Original', frame)
            # cv2.imshow('Single frame reference method', mask0.astype(float))
            cv2.imshow("Exponential update of reference frame method", mask1.astype(np.float64))
            # cv2.imshow("Median of last frames", mask2.astype(np.float64))

            key = cv2.waitKey(2)
            if key == ord('q'):
                break
            elif key == ord("p"):
                cv2.waitKey(-1)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    path = f"{DATASET_ROOT}/raw2/images/CV2021_GROUP05/group5.mp4"
    compare_methods(path)


if __name__ == "__main__":
    alpha = 0.33
    threshold = 52
    last_frames_number = 10

    main()
