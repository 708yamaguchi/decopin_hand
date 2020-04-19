#!/usr/bin/env python
# -*- coding: utf-8 -*-

# visualize created dataset
# You can view next image by pressing keys ('q' is quit)

import cv2
from cv_bridge import CvBridge
import numpy as np
import os.path as osp
import pickle
from process_gray_image import noise_subtract, smooth_gray_image, normalize_gray_image, img_jet # NOQA
import rospkg


def show_and_save(img, save_path):
    cv2.imshow('Processed spectrogram', img)
    cv2.waitKey(0)
    cv2.imwrite(save_path, img)
    print('Saved: {}'.format(save_path))


def process_example():
    """
    Show and save processed images after noise subtraction and smooth.
    input: pickle file which contains sensor_msgs/Image object of 32FC1 image.
    """
    # Find example directory
    rospack = rospkg.RosPack()
    example_dir = osp.join(rospack.get_path(
        'decopin_hand'), 'scripts', 'example')
    if not osp.exists(example_dir):
        print('scripts/example directory is not found.')
        exit()
    # Load spectrogram
    bridge = CvBridge()
    spectrogram_path = osp.join(example_dir, 'spectrogram.pickle')
    with open(spectrogram_path, mode='rb') as f:
        spectrogram = pickle.load(f)
    img = bridge.imgmsg_to_cv2(spectrogram)  # Expected as 32FC1
    # Load noise
    noise_path = osp.join(example_dir, 'noise.npy')
    noise = np.load(noise_path)

    # Show and save original image
    normalized_img = normalize_gray_image(img)
    show_and_save(cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR).astype(np.uint8),
                  osp.join(example_dir, 'raw.png'))
    show_and_save(img_jet(normalized_img),
                  osp.join(example_dir, 'raw_jet.png'))
    # Show and save noise subtracted image
    img = noise_subtract(img, noise)
    normalized_img = normalize_gray_image(img)
    show_and_save(cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR).astype(np.uint8),
                  osp.join(example_dir, 'noise_subtracted.png'))
    show_and_save(img_jet(normalized_img),
                  osp.join(example_dir, 'noise_subtracted_jet.png'))
    # Show and save smoothed image
    img = smooth_gray_image(img)
    normalized_img = normalize_gray_image(img)
    show_and_save(cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR).astype(np.uint8),
                  osp.join(example_dir, 'smoothed.png'))
    show_and_save(img_jet(normalized_img),
                  osp.join(example_dir, 'smoothed_jet.png'))


if __name__ == '__main__':
    process_example()
