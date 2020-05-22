#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import os
import os.path as osp
from process_gray_image import img_jet, normalize_gray_image
import rospkg


class VisualizeSpectrogramProcess(object):
    """
    This class visualizes spectrogram processing elements:
    1. Noise to be subtracted
    2. Raw spectrogram calculated from audio data
    3. Spectrogram subtracted by noise
    4. Spectrogram subtracted by noise with class label

    Note that original_spectrogram directory is created by action_saver.py
    This script depends on the directory structure
    """

    def __init__(self):
        rospack = rospkg.RosPack()
        parser = argparse.ArgumentParser()
        parser.add_argument('--height', type=int, default=200,
                            help='Height of visualized image')
        parser.add_argument('--width', type=int, default=200,
                            help='Width of visualized image')
        parser.add_argument('--target-class', '-t', type=str, required=True,
                            help='Class label to refer')
        args = parser.parse_args()
        self.height = args.height
        self.width = args.width
        self.target_class = args.target_class
        self.spectrogram_dir = osp.join(
            rospack.get_path('decopin_hand'), 'train_data',
            'original_spectrogram', self.target_class)
        self.noise_data_path = osp.join(self.spectrogram_dir, 'noise.npy')

    def visualize_noise(self, window_name):
        if osp.exists(self.noise_data_path):
            noise_data = np.load(self.noise_data_path)
            mean_spectrum = np.mean(noise_data, axis=0)
            mean_spectrum = normalize_gray_image(mean_spectrum)
            mono = cv2.resize(mean_spectrum,
                              (self.width, self.height))
            jet = img_jet(mono)
            cv2.imshow(window_name, jet)
            return jet
        else:
            print('Noise data is not found at {}'.format(self.noise_data_path))

    def visualize_spectrogram(self, file_path, window_name):
        img = cv2.imread(file_path)[:, :, 0]
        jet = img_jet(img)
        cv2.imshow(window_name, jet)
        return jet

    def main(self):
        # For all images in spectrogram_dir, show noise and spectrograms
        for f in os.listdir(v.spectrogram_dir):
            if not (f.startswith(self.target_class) and f.endswith('png')):
                continue
            # Noise
            window_name = 'Noise spectrogram'
            noise = v.visualize_noise(window_name)
            cv2.moveWindow(window_name, 100, 0)
            # Raw spectrogram
            window_name = 'Raw spectrogram'
            raw = v.visualize_spectrogram(
                osp.join(v.spectrogram_dir, 'raw',
                         osp.splitext(f)[0]+'_raw.png'),
                window_name)
            cv2.moveWindow(window_name, 500, 0)
            # Processed spectrogram
            window_name = 'Processed spectrogram'
            processed = v.visualize_spectrogram(
                osp.join(v.spectrogram_dir, f),
                window_name)
            cv2.moveWindow(window_name, 900, 0)
            # Show and Save
            key = cv2.waitKey(0)
            if key == ord('q'):
                exit()
            elif key == ord('s'):
                img_path = osp.join(v.spectrogram_dir, 'noise_spectrogram.png')
                cv2.imwrite(img_path, noise)
                print('Saved {}'.format(img_path))
                img_path = osp.join(v.spectrogram_dir, 'raw_spectrogram.png')
                cv2.imwrite(img_path, raw)
                print('Saved {}'.format(img_path))
                img_path = osp.join(v.spectrogram_dir, 'processed_spectrogram.png')
                cv2.imwrite(img_path, processed)
                print('Saved {}'.format(img_path))
            else:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    v = VisualizeSpectrogramProcess()
    v.main()
