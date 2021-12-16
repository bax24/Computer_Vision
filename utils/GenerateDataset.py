import os

import cv2
from tqdm import tqdm

import src
from utils import exists_or_mkdir, prepare_annotations


def generate_dataset(video_path, annotations_path, frame_starting_index, new_dataset_path):
    # Prepare folders
    exists_or_mkdir(new_dataset_path)
    exists_or_mkdir('{}/Annotations'.format(new_dataset_path))
    exists_or_mkdir('{}/Images'.format(new_dataset_path))

    # Video reader
    cap = cv2.VideoCapture(video_path)

    """
    Import annotations and transform in an array:
    Indexed by frame
    For each frame a dict
    Keys are trackers
    Values are [[drop coord], nb_cells, [[cells coord]]]
    Drop coord: [x_min, x_max, x_center, width]
    Cell coord: [x_center, y_center, width, height]"""
    annotations = prepare_annotations(annotations_path)

    # Reading loop
    output_idx = frame_starting_index
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(annotations):
            break

        # Grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get annotations
        data = annotations[frame_idx]

        if data is None:  # No droplet in the frame
            frame_idx += 1
            continue

        # Plot all droplets
        for idx in range(len(data.keys())):

            # Create an image with just the droplet
            key = list(data.keys())
            drp_coord = data[key[idx]][0]

            # Don't tale parts of the droplets
            if drp_coord[1] - drp_coord[0] < src.MINIMUM_DRP_WIDTH:
                break

            sub_frame = frame[0:240, drp_coord[0]:drp_coord[1]]
            sub_frame = cv2.resize(sub_frame, (240, 240), cv2.INTER_AREA)

            # Write to file
            cv2.imwrite('{}/Images/{}.jpg'.format(new_dataset_path, output_idx), sub_frame)

            # If cells in droplet -> add corrdinates in annotation file
            if data[key[idx]][1] > 0:
                cells_coord = data[key[idx]][2]
                f = open('{}/Annotations/{}.txt'.format(new_dataset_path, output_idx), 'w')
                for k in range(0, len(cells_coord)):
                    # First we need relative coordinates to the cell
                    x_cell = (cells_coord[k][0] - drp_coord[0]) / drp_coord[3]  # divided by the cell width
                    width_cell = cells_coord[k][2] / drp_coord[3]
                    f.write('0 {} {} {} {}\n'.format(
                        x_cell,
                        cells_coord[k][1] / 240,
                        width_cell,
                        cells_coord[k][3] / 240))
                f.close()
            output_idx += 1

        frame_idx += 1

    return output_idx


if __name__ == "__main__":
    folders_list = os.listdir("{}/raw2/images".format(src.DATASET_ROOT))

    start_idx = 0
    for i in tqdm(range(len(folders_list))):
        group_index = int(folders_list[i][-2:])
        video_path = "{}/raw2/images/{}/group{}.mp4".format(src.DATASET_ROOT, folders_list[i], group_index)
        annotations_path = "{}/raw2/annotations/{}/{}.csv".format(src.DATASET_ROOT, folders_list[i], folders_list[i])

        start_idx = generate_dataset(video_path, annotations_path, start_idx, src.MAIN_DATASET)
