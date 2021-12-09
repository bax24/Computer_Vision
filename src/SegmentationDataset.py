import os

import cv2
import numpy as np
import pandas as pd
import torch.utils.data
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import DATASET_ROOT

CELL_RADIUS = 20 / 2
DROPLET_RADIUS = 312 / 2
DROPLET_THICKNESS = 5


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True, test_size=0.2, transforms=None, seed=None):
        super(SegmentationDataset, self).__init__()
        self.transforms = transforms

        annotations_path = os.path.join(data_dir, "annotations")
        images_path = os.path.join(data_dir, "images")
        list_groups = os.listdir(images_path)

        # List all images and load them
        # self.images_path = []
        # for g in list_groups:
        #     csv_path = os.path.join(images_path, g)
        #     for img_path in os.listdir(csv_path):
        #         self.images_path.append(os.path.join(csv_path, img_path))

        # List all annotations and load them into pandas dataframe
        self.annotations = None
        for g in list_groups:
            csv_path = f"{annotations_path}/{g}.csv"
            annotations = pd.read_csv(csv_path, header=[0], sep=";")
            annotations["Filename"] = annotations["Slice"].apply(lambda x: f"{images_path}/{g}/frame_{x}.jpg")

            # Append annotations
            if self.annotations is None:
                self.annotations = annotations
            else:
                self.annotations = self.annotations.append(annotations, ignore_index=True)

        droplet_annotations = self.annotations.loc[lambda df: df["Label"] == "Droplet"]
        droplet_annotations["Width"] = droplet_annotations.apply(lambda df: df["XMax"] - df["XMin"], axis=1)
        droplet_annotations.drop(droplet_annotations[droplet_annotations["Width"] < 268].index,
                                 inplace=True)
        droplet_annotations.drop(droplet_annotations[droplet_annotations["Width"] > 400].index,
                                 inplace=True)
        droplet_annotations.drop(droplet_annotations[droplet_annotations["Slice"] < 400].index,
                                 inplace=True)

        train_ds, test_ds = train_test_split(droplet_annotations, test_size=test_size, random_state=seed)
        if train:
            self.droplet_annotations = train_ds
        else:
            self.droplet_annotations = test_ds
        self.counter = 0

    def __getitem__(self, idx):
        # img_path = self.images_path[idx]
        img_path = self.droplet_annotations.iloc[idx].Filename
        target = self.annotations.loc[self.annotations.Filename == img_path]
        image = np.array(Image.open(img_path).convert("L"))
        image, mask = get_data(image, target)

        if self.transforms is not None:
            return self.transforms(image, mask)

        return image, mask

    def __len__(self):
        # return len(self.images_path)
        return len(self.droplet_annotations)


def get_data(image, target):
    """
    Return only the mask of the cells
    """
    # ground_truth = np.zeros(shape, dtype=np.uint8)
    # droplet_mask = np.zeros(shape, dtype=np.uint8)
    cell_mask = np.zeros(image.shape, dtype=bool)

    xmin, ymin = np.inf, np.inf
    xmax, ymax = np.NINF, np.NINF

    droplet_seen = False
    for idx, row in target.iterrows():
        if row["Label"] == "Droplet" and droplet_seen is False:
            droplet_seen = True
            xmin = min(xmin, row["XMin"])
            xmax = max(xmax, row["XMax"])
            ymin = min(ymin, row["YMin"])
            ymax = max(ymax, row["YMax"])

        center_x = (row["XMin"] + row["XMax"]) / 2
        center_y = (row["YMin"] + row["YMax"]) / 2

        if row["Label"] == "Cell":
            mask = draw_mask(image.shape, (center_x, center_y), CELL_RADIUS)
            cell_mask[np.where(mask > 0)] = bool(1)
        # elif row["Label"] == "Droplet":
        #     center_x = check_border(center_x, row, shape)
        #     mask = draw_mask(shape, (center_x, center_y), DROPLET_RADIUS)
        #     droplet_mask[np.where(mask > 0)] = 1

    # ground_truth[np.where(droplet_mask > 0)] = 0
    # ground_truth[np.where(cell_mask > 0)] = 1

    image = image[ymin:ymax, xmin:xmax]
    cell_mask = cell_mask[ymin:ymax, xmin:xmax]

    return image, cell_mask


def check_border(x, row, img_shape):
    if x - DROPLET_RADIUS < 0:
        x = row["XMax"] - DROPLET_RADIUS
    elif x + DROPLET_RADIUS > img_shape[1]:
        x = row["XMin"] + DROPLET_RADIUS

    return x


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
    # generate_dataset()

    d = SegmentationDataset(os.path.join(DATASET_ROOT, "clean"), train=False, test_size=0.3)

    key = ord("a")
    while key != ord("q"):
        img, mask = d[np.random.randint(len(d))]
        mask = mask.astype(np.uint8) * 255

        cv2.imshow("Image", img)
        cv2.imshow('Mask', mask)
        cv2.waitKey(-1)
        key = cv2.waitKey(10)
