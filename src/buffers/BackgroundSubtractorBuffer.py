import time
from queue import Queue
from threading import Thread


class BackgroundSubtractorBuffer:

    def __init__(self,
                 input_buffer,
                 background_subtractor,
                 buff_size=500
                 ):

        # Store the input buffers
        self.input = input_buffer

        # Function to apply on the frames
        self.background_subtractor = background_subtractor

        # The output buffers
        self.buffer = Queue(maxsize=buff_size)

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
                frame, frame_idx = self.input.read()

                mask = self.background_subtractor(frame)
                # Append in the buffers
                self.buffer.put((frame, mask, frame_idx))

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
