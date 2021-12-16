import time
from queue import Queue
from threading import Thread

import cv2
import numpy as np

COLOR = (255, 0, 0)


class DropletExtractorBuffer:

    def __init__(self,
                 input_buffer,
                 buff_size=500,
                 droplet_threshold=30,
                 droplet_min_width=250,
                 ):

        # Store the input buffers
        self.input = input_buffer

        # The output buffers
        self.buffer = Queue(maxsize=buff_size)

        # Coordinate extraction params
        self.droplet_threshold = droplet_threshold
        self.droplet_min_width = droplet_min_width

        # Thread management
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.end = False
        self.pred_end = False

    def start(self):

        self.thread.start()
        return self

    def update(self):

        while self.input.more():

            # Perform iterations to fill the buffers
            if not self.buffer.full():

                # Get a frame
                frame, mask, frame_idx = self.input.read()
                p1, p2 = self.extract_coordinates(mask)

                if (p1, p2) == ((0, 0), (0, 0)):
                    droplet = None
                else:
                    droplet = frame[p1[1]:p2[1], p1[0]:p2[0]]
                    frame = cv2.rectangle(frame, p1, p2, color=COLOR, thickness=3)

                # Append in the buffers
                self.buffer.put((frame, mask, droplet, frame_idx))

            else:
                # If the buffers is already full, wait 0.01 sec
                time.sleep(0.01)

        # When the previous algorithm end
        self.end = True

    def read(self):
        """
        :return: The oldest tuple in the buffers:
        (<original frame>, <the MOG mask>, <the frame index>)
        """
        return self.buffer.get()

    def more(self):
        """
        If the buffers is empty but not already set as
        end of video, we continue 20 times to wait a
        potential refilling
        :return: True if there are elements after waiting
        """

        while self.buffer.qsize() == 0 and not self.end:
            time.sleep(0.01)

        buff_fill = False
        if self.buffer.qsize() > 0:
            buff_fill = True

        return buff_fill

    def running(self):

        if self.end:
            return False
        else:
            return self.more()

    def stop(self):

        self.end = True
        self.thread.join()

    def extract_coordinates(self, mask):

        col_sum = mask.sum(axis=0) / 255
        hit = (col_sum >= self.droplet_threshold)
        droplet_index = np.argwhere(hit != 0)

        if len(droplet_index) == 0:
            # Frame without any droplet
            return (0, 0), (0, 0)

        starting_coord = min(droplet_index)[0]
        ending_coord = max(droplet_index)[0]

        if ending_coord - starting_coord < self.droplet_min_width:
            return (0, 0), (0, 0)

        upper_left_point = (starting_coord, 0)
        lower_right_point = (ending_coord, mask.shape[1])
        return upper_left_point, lower_right_point
