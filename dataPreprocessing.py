
import cv2
import numpy as np


def preprocess(img):
    """
    Preprocess a BGR/RGB image for the Nvidia CNN model.
    Steps: crop → YUV → Gaussian blur → resize (200×66) → normalize
    Returns float32 array of shape (66, 200, 3) with values in [0, 1].
    """
    img = img[60:135, :, :]                          # crop to road area
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)       # BGR→YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)           # reduce noise
    img = cv2.resize(img, (200, 66))                 # resize to model input
    img = (img / 255).astype(np.float32)             # normalize to [0, 1]
    return img
