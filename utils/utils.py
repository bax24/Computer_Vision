import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from src import DATASET_ROOT


def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_videos_path():
    raw_dataset = os.path.join(DATASET_ROOT, "raw2")
    groups_name = os.listdir(os.path.join(raw_dataset, "images"))

    video_list = []
    for group in groups_name:
        group_idx = int(group[-2:])
        video_path = "{}/images/{}/group{}.mp4".format(raw_dataset, group, group_idx)
        video_list.append(video_path)

    return video_list


def prepare_annotations(path):
    """
        Load the annotation csv and sort it
        """
    annot = pd.read_csv(path, sep=';')

    # First we sort by frame index (slice)
    annot = annot.sort_values(by=['Slice'], ascending=True, ignore_index=True)

    # Get the max slice index
    max_slice = annot['Slice'].max()

    # Build an array indexed by slices and who condain all bboxes
    data = []
    for i in range(0, max_slice):
        # Select elements with this slice
        df = annot[annot['Slice'] == i]

        # If no box in this slice
        if df.shape[0] == 0:
            data.append(None)
            continue

        # A dictionary: Keys = tracker of the droplet, Values = [droplet_coord, nb_cells, [cells coord]]
        sub_data = {}
        # Read all droplets in this frame
        for j in range(0, df.shape[0]):

            # Don't care about cells in this loop
            if not 'Droplet' in df.iloc[j]['Terms']:
                continue
            if 'POINT' in df.iloc[j]['Geometry']:
                continue

            # Get Tracker index
            try:
                tracker = int(str(df.iloc[j]['Track']).replace('[', '').replace(']', ''))
            except:
                # In some cases we have a droplet without tracker
                continue

            # Get droplet coordinates and clean it
            coord = df.iloc[j]['Geometry']
            coord = coord.replace('POLYGON ((', '')
            coord = coord.replace('))', '')
            coord = coord.replace(',', '')
            coord = coord.split(' ')

            tl_x = float(coord[0])  # Top left x
            tl_y = float(coord[1])  # Top left y
            tr_x = float(coord[2])  # Top right x
            tr_y = float(coord[3])  # Top right y
            br_x = float(coord[4])  # Bottom right x
            br_y = float(coord[5])  # ...
            bl_x = float(coord[6])
            bl_y = float(coord[7])

            x_l = min(tl_x, tr_x, br_x, bl_x)
            x_r = max(tl_x, tr_x, br_x, bl_x)

            # Our general box format for droplets: (start_x, end_x, center_x, width)
            center_x = int((x_r + x_l) / 2)
            width = int(x_r - x_l)
            drp_coord = [int(x_l), int(x_r), center_x, width]
            if drp_coord[0] < 0:
                drp_coord[0] = 0
            if drp_coord[1] >= 1600:
                drp_coord[1] = 1599
            sub_data[str(tracker)] = [drp_coord, 0, []]

        # Now we look at cell's
        for j in range(0, df.shape[0]):

            # Don't care about Droplets in this loop
            if not 'Cell' in df.iloc[j]['Terms']:
                continue

            # Get Cell coordinates and clean it
            coord = df.iloc[j]['Geometry']
            coord = coord.replace('POLYGON ((', '')
            coord = coord.replace('))', '')
            coord = coord.replace(',', '')
            coord = coord.split(' ')
            tl_x = float(coord[0])  # Top left x
            tl_y = float(coord[1])  # Top left y
            tr_x = float(coord[2])  # Top right x
            tr_y = float(coord[3])  # Top right y
            br_x = float(coord[4])  # Bottom right x
            br_y = float(coord[5])  # ...
            bl_x = float(coord[6])
            bl_y = float(coord[7])

            x_l = min(tl_x, tr_x, br_x, bl_x)
            x_r = max(tl_x, tr_x, br_x, bl_x)
            y_b = max(tl_y, tr_y, br_y, bl_y)
            y_t = min(tl_y, tr_y, br_y, bl_y)

            # General cell box format: (x_center, y_center, width, height)
            x_center = int((x_r + x_l) / 2)
            y_center = int((y_b + y_t) / 2)
            width = int(x_r - x_l)
            height = int(y_b - y_t)
            crd = [x_center, y_center, width, height]

            # Check in which droplet it come
            done = False
            for key in sub_data.keys():
                if x_center < sub_data[key][0][1] and x_center > sub_data[key][0][0]:
                    done = True
                    sub_data[key][1] += 1
                    sub_data[key][2].append(crd)
                    break
            if not done:
                # print('WARNING: can not find a droplet for slice {}, cell {}'.format(i, crd))
                pass

        data.append(sub_data)

    return data


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


def count_peaks_2d(matrix, pk_min_thresh=5):
    """
    This function is an adaptation of the
    cout_peaks_2d_one_dim method. In this
    implementation, predictions are done
    a first time by doing the sum of columns
    before analyzing detected columns,
    and a second time by starting from the sum
    over rows.
    This two times prediction checking permit
    us to analyze from another point of view
    who may sometime detect cells difficulties
    detectable in the other.
    """
    debug = False
    # Horizontal prediction:
    counter_h = 0
    results_h = []

    # Makes the sum over columns
    col_sum = np.sum(matrix, axis=0)

    # Find peaks index
    peaks, _ = find_peaks(col_sum, distance=5)

    # Check if multiple cell in this peak
    for pk in peaks:
        if debug:
            print('------------')
            print('col_sum peaks: {}'.format(pk))
        # Check if the peak is greater than the threshold
        if col_sum[pk] >= pk_min_thresh:
            sub_peaks, _ = find_peaks(matrix[:, pk], distance=5)
            for sub_pk in sub_peaks:
                if matrix[sub_pk, pk] > 1.5:
                    counter_h += 1
                    results_h.append([pk, sub_pk])
            if debug:
                plt.plot(matrix[:, pk], color='red')
                plt.title('Sub peak {}'.format(pk))
                plt.show()

    # Vertical prediction:
    counter_v = 0
    results_v = []
    line_sum = np.sum(matrix, axis=1)
    peaks, _ = find_peaks(line_sum, distance=5)
    for pk in peaks:
        # Check if the peak is greater than the threshold
        if line_sum[pk] >= pk_min_thresh:
            sub_peaks, _ = find_peaks(matrix[pk, :], distance=5)
            for sub_pk in sub_peaks:
                if matrix[pk, sub_pk] > 1.5:
                    counter_v += 1
                    results_v.append([sub_pk, pk])

    if debug:
        plt.plot(col_sum)
        plt.show()
        plt.close()
    if counter_h >= counter_v:
        return counter_h, results_h
    return counter_v, results_v


def shuffle_lst(lst, seed=1):
    """
    Simple code to shuffle a list
    with a fixed seed
    """
    np.random.seed(seed)
    idx_lst = np.arange(len(lst))
    np.random.shuffle(idx_lst)

    return [lst[i] for i in idx_lst]


def heat_map_gen(coordinates, img_shape):
    """
    This code generate a 240 x 240 px heat-
    map to produce target images during the
    Unet training process.
    Heat maps contain a 2D gaussian
    centered on each cell center coordinates
    such that the sum of the values in the
    heatmap is equal to 100 * the number of
    cells in the image.
    """

    # Init the array
    label = np.zeros(img_shape, dtype=np.float32)

    # If no annotations (so no cell)
    if len(coordinates) == 0:
        return label

    # Map the center of each object
    for x, y in coordinates:
        if x >= 240:
            x = 239
        if y >= 240:
            y = 239
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        label[int(y), int(x)] = 100

    # Apply a gaussian filter convolution
    label = gaussian_filter(label, sigma=(1, 1), order=0)

    return label


def avg_smoothing(seq, window_size):
    start_idx = 0
    end_idx = window_size
    target_idx = int(window_size / 2)
    output = np.zeros(len(seq) + window_size)
    for i in range(int(window_size / 2)):
        output[i] = seq[0]
        output[-i] = seq[-1]
    output[int(window_size / 2):int(window_size / 2) + len(seq)] = seq

    while end_idx < len(output):
        output[target_idx] = np.mean(seq[start_idx:end_idx])
        start_idx += 1
        end_idx += 1
        target_idx += 1

    return output[int(window_size / 2): int(window_size / 2) + len(seq)]
