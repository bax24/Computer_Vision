import time
from queue import Queue
from threading import Thread

import cv2


class VideoReader:
    def __init__(self,
                 path_list,
                 buff_size=500):

        # Index in the path list
        self.path_idx = 0
        self.path_list = path_list

        # The video reading object
        self.vid = cv2.VideoCapture(path_list[0])
        # The buffers
        self.buffer = Queue(maxsize=buff_size)

        # Thread management
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.end = False

        # Count the frames
        self.frame_idx = 0

    def start(self):

        self.thread.start()
        return self

    def update(self):

        while not self.end:

            # Fill the buffers
            if not self.buffer.full():
                ret, frame = self.vid.read()

                if not ret:
                    # End of the files to read
                    if self.path_idx + 1 >= len(self.path_list):
                        self.end = True
                        break
                    # If there are remaining files to read
                    else:
                        self.path_idx += 1
                        self.vid.release()
                        self.vid = cv2.VideoCapture(self.path_list[self.path_idx])
                        continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.buffer.put((frame, self.frame_idx))
                self.frame_idx += 1

            else:
                # If the buffers is already full, wait 0.1 sec
                time.sleep(0.1)

        # Close the video reader
        self.vid.release()

    def read(self):
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
