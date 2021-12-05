import os

import cv2


def draw_rectangle(frame, df, slice_number):
    df_data = df.loc[df.Slice == slice_number]
    droplet_color = (255, 123, 0)
    cell_color = (123, 0, 255)

    if "Cell" in df_data.values:
        cells = df_data.loc[df_data.Type == "Cell"]
        for idx, row in cells.iterrows():
            cv2.rectangle(frame, row.Points[0], row.Points[2], cell_color, 2)

    if "Droplet" in df_data.values:
        points = df_data.loc[df_data.Type == "Droplet"]["Points"].item()
        cv2.rectangle(frame, points[0], points[2], droplet_color, 3)


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)


def open_or_exit(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Error while opening video file! \n")
        exit(-1)
    else:
        return cap
