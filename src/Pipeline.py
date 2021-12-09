import cv2
import numpy as np
import torch

from src.config import DATASET_ROOT
from src.models.GaussianDensityEstimation import GaussianDensityEstimation
from src.models.UNet import UNet
from utils.FrameRate import FrameRate
from utils.utils import open_or_exit


def cleaner(frame, kernel, iterations):
    frame = cv2.erode(frame, kernel, iterations)
    frame = cv2.dilate(frame, kernel, iterations)
    return frame


def crop(frame, foreground, min_value):
    hits = foreground.sum(axis=0) >= min_value
    indexes = np.argwhere(hits != 0)
    if len(indexes) == 0:
        return None

    box_start = min(indexes)[0]
    box_end = max(indexes)[0]
    if box_end - box_start < 268:
        return None

    return (box_start, box_end), frame[:, box_start:box_end]


def reshape_tensor(img, size):
    img = np.resize(img, size)
    img = torch.tensor(np.array([img]), dtype=torch.float32)
    img = img.expand((1, img.shape[0], img.shape[1], img.shape[2]))
    return img


class Pipeline:
    def __init__(self, background_substractor, cell_detector, frame_rate, input_size=(64, 64)):
        self.background_substractor = background_substractor
        self.input_size = input_size

        self.frame_rate = frame_rate
        self.to_gray = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cell_detector = cell_detector
        self.cell_detector.to(self.device)
        self.cell_detector.eval()

    def __call__(self, video_cap):
        video_cap.set(2, 0.0)
        while video_cap.isOpened():
            ret, frame = video_cap.read()

            if ret:
                frame = self.to_gray(frame)
                foreground = self.background_substractor(frame) / 255  # {0, 1}
                foreground = cleaner(foreground, (20, 20), 5)

                crop_return = crop(frame, foreground, 30)

                predictions = None
                if crop_return is not None:
                    (bbox_start, bbox_end), roi = crop_return

                    roi_dim = (frame.shape[0], bbox_end - bbox_start)

                    roi = reshape_tensor(roi, self.input_size)
                    predictions = self.get_prediction(roi)
                    predictions = np.resize(predictions, roi_dim)

                self.frame_rate.update()
                if predictions is not None:
                    cv2.imshow("Predicted cells", predictions)

                cv2.imshow("Frame", frame)

                key = cv2.waitKey(5)
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.waitKey(-1)
            else:
                break

        video_cap.release()
        cv2.destroyAllWindows()

    def get_prediction(self, roi):
        roi = roi.to(self.device)
        with torch.no_grad():
            predictions = torch.sigmoid(self.cell_detector(roi).to("cpu"))

        predictions = torch.round(predictions)
        predictions = np.array(predictions[0, 0, :, :], dtype=np.uint8)

        return predictions * 255


frame_scale = 0.25
thresh = 1E-50
memory_size = 10
num_workers = 10

video_path = f"{DATASET_ROOT}/raw2/images/CV2021_GROUP11/group11.mp4"
video_cap = open_or_exit(video_path)

_, frame = video_cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

GDE = GaussianDensityEstimation(
    memory_size, frame, frame_scale=frame_scale, num_workers=num_workers, num_chunks=num_workers * 10)
GDE.fill_memory(video_cap)

model = UNet(1, 1, features=[64, 128, 256])
model.load_from_checkpoint(
    "/home/fares/PycharmProjects/Computer_Vision/models/sample-v2-epoch=02-val_loss=0.00000.ckpt")

frame_rate = FrameRate(verbose=False, frame_rate=5)
pipeline = Pipeline(GDE, model, frame_rate)
pipeline(video_cap)

# video_cap.set(2, 0.0)
# while video_cap.isOpened():
#     ret, frame = video_cap.read()
#
#     if ret:
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         logits = GDE(frame)
#         hits = (logits <= thresh).astype(np.uint8)
#         hits = cv2.dilate(cv2.erode(hits, erosion_kernel, num_iters), erosion_kernel, num_iters)
#         roi = crop_on_hits(frame, hits, 30)
#
#         frame_rate.update()
#         cv2.imshow("Frame", frame)
#         if roi is not None:
#             roi = reshape_tensor(roi[1])
#             roi = roi.expand((1, roi.shape[0], roi.shape[1], roi.shape[2])).to("cuda:0")
#             with torch.no_grad():
#                 prediction = torch.sigmoid(model(roi).to("cpu"))
#
#             prediction = np.array(prediction, dtype=np.uint8)
#             cv2.imshow("ROI", prediction[0, 0, :, :] * 255)
#
#         key = cv2.waitKey(5)
#         if key == ord('q'):
#             break
#         elif key == ord('p'):
#             cv2.waitKey(-1)
#     else:
#         break
#
# video_cap.release()
# cv2.destroyAllWindows()
