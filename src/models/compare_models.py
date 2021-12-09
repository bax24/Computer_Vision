from BackgroundSubstractor import *
from GaussianDensityEstimation import GaussianDensityEstimation
from src.config import DATASET_ROOT
from utils.FrameRate import FrameRate
from utils.utils import open_or_exit


def process_probs(probs):
    mask = (probs <= thresh).astype(np.uint8)
    mask *= 255
    return mask


def extract_bbox(frame, mask, threshold):
    hit = mask.sum(axis=0) / 255 >= threshold

    hit_indexes = np.argwhere(hit != 0)
    hit = np.array([hit] * frame.shape[0], dtype=np.uint8) * 255
    if len(hit_indexes) == 0:
        return frame, hit

    start_box = min(hit_indexes)[0]
    end_box = max(hit_indexes)[0]
    if end_box - start_box < 268:
        return frame, hit

    p1 = (start_box, 0)
    p2 = (end_box, frame.shape[1])
    frame = cv2.rectangle(frame, p1, p2, (255, 31, 42), 4)

    return frame, hit


if __name__ == "__main__":
    path = f"{DATASET_ROOT}/raw2/images/CV2021_GROUP11/group11.mp4"
    cap = open_or_exit(path)

    frame_scale = 0.5
    thresh = 1E-50
    memory_size = 15
    num_workers = 8

    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # GMM = GaussianMixtureModel(memory_size, frame, frame_scale=frame_scale, num_workers=num_workers,
    #                            num_chunks=num_workers * 10)
    # GMM.fill_memory(cap)

    GDE = GaussianDensityEstimation(memory_size, frame, frame_scale=frame_scale, num_workers=num_workers,
                                    num_chunks=num_workers * 10)
    GDE.fill_memory(cap)

    # Reset video head
    cap.set(2, 0.0)

    frame_rate = FrameRate(verbose=True, frame_rate=10)
    while cap.isOpened():

        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if ret:
            # GMM_mask = remove_noise(process_probs(GMM(img)), (10, 10), 10)
            GDE_mask = remove_noise(process_probs(GDE(img)), (10, 10), 10)

            frame_rate.update()
            # cv2.imshow("Original frame", img)
            # cv2.imshow("GMM Mask", GMM_mask)

            frame, hit = extract_bbox(img, GDE_mask, 30)
            cv2.imshow("Frame", frame)
            cv2.imshow("Hit", hit)
            cv2.imshow("GDE Mask", GDE_mask)

            key = cv2.waitKey(5)
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(-1)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
