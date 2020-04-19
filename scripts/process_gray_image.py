import cv2
import matplotlib.cm as cm
import numpy as np
from PIL import Image as Image_


def noise_subtract(img, noise):
    spectrogram_subtracted = img.transpose() - noise
    # Spectral subtraction method
    spectrogram_subtracted = np.where(spectrogram_subtracted > 0,
                                      spectrogram_subtracted,
                                      noise * 0.01)
    spectrogram_subtracted = spectrogram_subtracted.transpose()
    return spectrogram_subtracted


def smooth_gray_image(raw_img):
    """
    Blur to gray image
    input:  cv2 image, 32FC1
    output: cv2 image, 32FC1
    """
    return cv2.blur(raw_img, (5, 5))


def normalize_gray_image(mono_32fc1):
    """
    Convert input gray image to 8FC1 gray image.
    At this time, each pixel is normalized between 0 ~ 255.
    input:  cv2 image, 32FC1
    output: cv2 image, 8FC1
    """
    _max = mono_32fc1.max()
    _min = mono_32fc1.min()
    mono_32fc1 = (mono_32fc1 - _min) / (_max - _min) * 255.0
    mono_8uc1 = mono_32fc1.astype(np.uint8)
    return mono_8uc1


def img_jet(im):
    """
    Convert input to jet image if input is mono image.
    input : PIL 8UC1 image
    output: PIL 8UC3 image
    """
    img = np.array(im)
    if len(img.shape) == 2:
        normalized_img = img / 255.0
        jet = np.array(cm.jet(1 - normalized_img)[:, :, :3] * 255, np.uint8)
        jet = jet[:, :, [2, 1, 0]]  # bgr -> rgb
    else:
        jet = im
    return Image_.fromarray(jet)
