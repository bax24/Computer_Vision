import os
import time

import cv2
import pandas as pd

from config import DATASET_ROOT, FRAME_RATE
from dataset import extract_annotations
from utils import draw_rectangle


# TODO : this method works only with raw2 dataset -> make it use the clean dataset
def display_video_with_mask():
    group = "CV2021_GROUP01"

    video_path = os.path.join(DATASET_ROOT, "raw2/images/{}/{}.mp4".format(group, "group1"))
    annotation_path = os.path.join(DATASET_ROOT, "raw2/annotations/{}/{}.csv".format(group, group))

    df_raw_annotations = pd.read_csv(annotation_path, header=[0], sep=";")
    df_annotations = extract_annotations(df_raw_annotations)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error while opening file! \n")
        exit(-1)

    prev = 0
    frame_counter = 0

    while frame_counter < 385:
        frame_counter += 1
        _, _ = cap.read()

    while cap.isOpened():
        time_elapsed = time.time() - prev
        if time_elapsed <= 1. / FRAME_RATE:
            continue

        prev = time.time()
        ret, frame = cap.read()
        if ret:

            if frame_counter in df_annotations.values:
                draw_rectangle(frame, df_annotations, frame_counter)

            cv2.imshow("Frame", frame)

            frame_counter += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# TODO: Debug
if __name__ == "__main__":
    display_video_with_mask()
