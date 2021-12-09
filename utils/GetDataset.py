import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import DATASET_ROOT
from utils.utils import mkdir_if_not_exist


def extract_points(raw_data):
    raw_data = raw_data.split(" ")[1:-2]
    raw_data[0] = raw_data[0][2:]
    raw_data = [int(float(item[:-1])) if item.endswith(",") else int(float(item)) for item in raw_data]
    raw_data = [0 if item < 0 else item for item in raw_data]

    x_min = np.Inf
    x_max = np.NINF
    y_min = np.Inf
    y_max = np.NINF
    for idx in range(4):
        x_max = max(x_max, raw_data[2 * idx])
        y_max = max(y_max, raw_data[2 * idx + 1])
        x_min = min(x_min, raw_data[2 * idx])
        y_min = min(y_min, raw_data[2 * idx + 1])

    return [x_min, x_max, y_min, y_max]


def extract_annotations(raw_annotations):
    dataset = pd.DataFrame(columns=["Slice", "Label", "Confidence", "XMin", "XMax", "YMin", "YMax"])
    for idx, row in raw_annotations.iterrows():
        if row["Geometry"].split(" ")[0] == "POINT":
            continue
        points = extract_points(row["Geometry"])
        label = row["Terms"].split(" ")[-1]

        data = [row["Slice"], label, 1] + points
        dataset.loc[idx] = data

    dataset.sort_values(by=["Slice"], inplace=True, ascending=True)
    return dataset


def generate_dataset():
    raw_dataset = os.path.join(DATASET_ROOT, "raw2")
    group_names = os.listdir(os.path.join(raw_dataset, "images"))
    new_dataset = os.path.join(DATASET_ROOT, "clean")
    mkdir_if_not_exist(new_dataset)

    annotations_path = os.path.join(new_dataset, "annotations")
    images_path = os.path.join(new_dataset, "images")
    mkdir_if_not_exist(annotations_path)
    mkdir_if_not_exist(images_path)

    for group in tqdm(group_names):
        # Extract the annotations
        _annotation_path = os.path.join(raw_dataset, "annotations/{}/{}.csv".format(group, group))
        df_raw_annotations = pd.read_csv(_annotation_path, header=[0], sep=";")
        df_annotations = extract_annotations(df_raw_annotations)
        # noinspection PyTypeChecker
        df_annotations.to_csv(f'{annotations_path}/{group}.csv', index=False, sep=";")

        # Extract the images
        group_idx = int(group[-2:])
        _video_path = "{}/images/{}/group{}.mp4".format(raw_dataset, group, group_idx)
        _images_path = os.path.join(images_path, group)
        mkdir_if_not_exist(_images_path)

        cap = cv2.VideoCapture(_video_path)
        if not cap.isOpened():
            print("Error while opening file! \n")
            exit(-1)

        frame_counter = 0
        while frame_counter < 400:
            _, _ = cap.read()
            frame_counter += 1

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(_images_path, "frame_{}.jpg".format(frame_counter)), frame)
                frame_counter += 1
            else:
                cap.release()


if __name__ == "__main__":
    generate_dataset()
