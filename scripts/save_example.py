#!/usr/bin/env python

import cv2
import numpy as np
import os.path as osp
import pickle
from sensor_msgs.msg import Image
import rospkg
import rospy


# rosrun decopin_hand save_example input:=/spectrum_to_spectrogram/spectrogram
class SaveExample(object):

    def __init__(self):
        rospack = rospkg.RosPack()
        train_dir = osp.join(rospack.get_path(
            'decopin_hand'), 'train_data')
        example_dir = osp.join(rospack.get_path(
            'decopin_hand'), 'scripts', 'example')
        # Load noise
        original_noise_path = osp.join(train_dir, 'noise.npy')
        if osp.exists(original_noise_path):
            noise_data = np.load(original_noise_path)
            self.noise = np.mean(noise_data, axis=0)
            self.noise_path = osp.join(example_dir, 'noise.npy')
        else:
            rospy.logerr('{} is not found.'.format(original_noise_path))
            exit()
        # Load spectrogram
        self.spectrogram_path = osp.join(example_dir, 'spectrogram.pickle')

    def save_example(self):
        rospy.loginfo('Press key when you want to save spectrogram and noise.')
        cv2.waitKey(0)
        np.save(self.noise_path, self.noise)
        spectrogram = rospy.wait_for_message('input', Image)
        with open(self.spectrogram_path, mode='wb') as f:
            pickle.dump(spectrogram, f, protocol=2)
        rospy.loginfo('Successfully saved spectrogram and noise.')


if __name__ == '__main__':
    rospy.init_node('save_example')
    se = SaveExample()
    se.save_example()
