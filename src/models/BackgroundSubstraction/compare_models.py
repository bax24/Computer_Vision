from BackgroundSubstractor import *
from GaussianDensityEstimation import GaussianDensityEstimation
from src.config import DATASET_ROOT
from utils.FrameRate import FrameRate
from utils.utils import open_or_exit


def process_probs(probs):
    mask = (probs <= thresh).astype(np.uint8)
    mask *= 255
    return mask


if __name__ == "__main__":
    path = f"{DATASET_ROOT}/raw2/images/CV2021_GROUP11/group11.mp4"
    print(path)
    cap = open_or_exit(path)

    frame_scale = 0.66
    thresh = 1E-50
    memory_size = 10
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

    frame_rate = FrameRate(verbose=True, frame_rate=250)
    frame_counter = 0
    while cap.isOpened():

        ret, img = cap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if ret:
            # GMM_mask = remove_noise(process_probs(GMM(img)), (10, 10), 10)
            GDE_mask = remove_noise(process_probs(GDE(img)), (10, 10), 10)

            frame_rate.update()
            cv2.imshow("Original frame", img)
            # cv2.imshow("GMM Mask", GMM_mask)
            cv2.imshow("GDE Mask", GDE_mask)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey(-1)

            break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()
