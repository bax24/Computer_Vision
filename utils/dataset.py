import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.config import DATASET_ROOT
from utils import mkdir_if_not_exist

CELL_RADIUS = 31 / 2
DROPLET_RADIUS = 312 / 2
DROPLET_THICKNESS = 5


#############################
# Methods used to generate the
# clean dataset from the raw dataset
#############################
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
        _video_path = os.path.join(DATASET_ROOT, "images/{}/group{}.mp4".format(group, group_idx))
        _images_path = os.path.join(images_path, group)
        mkdir_if_not_exist(_images_path)

        cap = cv2.VideoCapture(_video_path)
        if not cap.isOpened():
            print("Error while opening file! \n")
            exit(-1)

        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(os.path.join(_images_path, "frame_{}.jpg".format(frame_counter)), frame)
                frame_counter += 1
            else:
                cap.release()


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, version=0):
        super(Dataset, self).__init__()
        self.transforms = transforms
        self.version = version

        annotations_path = os.path.join(root, "annotations")
        images_path = os.path.join(root, "images")
        groups = os.listdir(images_path)

        self.images = []
        for g in groups:
            tmp_path = os.path.join(images_path, g)
            for image_path in os.listdir(tmp_path):
                self.images.append(os.path.join(tmp_path, image_path))

        self.annotations = None
        for g in groups:
            tmp_path = f"{annotations_path}/{g}.csv"
            annotations = pd.read_csv(tmp_path, header=[0], sep=";")
            annotations["Filename"] = annotations["Slice"].apply(lambda x: f"{images_path}/{g}/frame_{x}.jpg")

            if self.annotations is None:
                self.annotations = annotations
            else:
                self.annotations = self.annotations.append(annotations, ignore_index=True)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        target = self.annotations.loc[self.annotations.Filename == img_path]
        img = np.array(Image.open(img_path).convert("L"))

        if self.version == 0:
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target

        elif self.version == 1:
            return img, get_mask(img.shape, target)

    def __len__(self):
        return len(self.images)


def get_mask(shape, target):
    ground_truth = np.zeros(shape, np.uint8)
    cell_mask = np.zeros(shape, np.uint8)
    droplet_mask = np.zeros(shape, np.uint8)

    for idx, row in target.iterrows():
        center_x = (row["XMin"] + row["XMax"]) / 2
        center_y = (row["YMin"] + row["YMax"]) / 2
        if row["Label"] == "Cell":
            mask = draw_mask(shape, (center_x, center_y), CELL_RADIUS)
            cell_mask[np.where(mask > 0)] = 1
        elif row["Label"] == "Droplet":
            mask = draw_mask(shape, (center_x, center_y), DROPLET_RADIUS)
            droplet_mask[np.where(mask > 0)] = 1

    ground_truth[np.where(droplet_mask > 0)] = 1
    ground_truth[np.where(cell_mask > 0)] = 2

    return ground_truth


def draw_mask(shape, center, radius):
    y, x = np.ogrid[:shape[0], :shape[1]]
    center_dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return center_dist <= radius


def get_dataset_stats(annotations):
    # Stats applied on droplets
    droplets_stats = pd.DataFrame()
    droplets_stats["Width"] = annotations.loc[annotations["Label"] == "Droplet"].apply(
        lambda row: row["XMax"] - row["XMin"], axis=1)
    droplets_stats["Height"] = annotations.loc[annotations["Label"] == "Droplet"].apply(
        lambda row: row["YMax"] - row["YMin"], axis=1)

    droplets_stats.drop(droplets_stats[droplets_stats["Width"] < 268].index, inplace=True)

    # Stats applied on cells
    cells_stats = pd.DataFrame()
    cells_stats["Width"] = annotations.loc[annotations["Label"] == "Cell"].apply(
        lambda row: row["XMax"] - row["XMin"], axis=1)
    cells_stats["Height"] = annotations.loc[annotations["Label"] == "Cell"].apply(
        lambda row: row["YMax"] - row["YMin"], axis=1)

    print("=== Droplets ===")
    print("- Mean -\n", droplets_stats.mean(axis=0))
    print("\n- Std -\n", droplets_stats.std(axis=0))

    print("\n=== Cells ===")
    print("- Mean -\n", cells_stats.mean(axis=0))
    print("\n- Std -\n", cells_stats.std(axis=0))


if __name__ == "__main__":
    d = Dataset(os.path.join(DATASET_ROOT, "clean"), None, version=1)
    # get_dataset_stats(d.annotations)

    key = ord("a")
    while key != ord("q"):
        img, mask = d.__getitem__(np.random.randint(len(d)))
        mask = mask * 127

        cv2.imshow("Image", img)
        cv2.imshow('Mask', mask)
        cv2.waitKey(-1)
        key = cv2.waitKey(10)
