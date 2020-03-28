#!/usr/bin/env python

import numpy as np
from os import path as osp

from cv_bridge import CvBridge
import rospkg
import rospy
from sensor_msgs.msg import Image


class NoiseSubtractor(object):
    """
    Subscribe raw spectrum from microphone
    Publish spectrum which is subtracted by noise spectrum
    """

    def __init__(self):
        rospack = rospkg.RosPack()
        self.train_dir = osp.join(rospack.get_path(
            'decopin_hand'), 'train_data')
        self.noise_data_path = osp.join(self.train_dir, 'noise.npy')
        if not osp.exists(self.noise_data_path):
            rospy.logwarn('{} is not found. Exit.'.format(self.noise_data_path))
            exit()
        noise_data = np.load(self.noise_data_path)
        self.mean_spectrum = np.mean(noise_data, axis=0)
        # ROS
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('~raw_spectrogram', Image, self.cb)
        self.pub = rospy.Publisher('~subtracted_spectrogram', Image, queue_size=1)

    def cb(self, msg):
        """
        Main process of NoiseSubtractor class
        Publish spectrum which is subtracted by noise spectrum
        """
        spectrogram_raw = self.bridge.imgmsg_to_cv2(msg)
        spectrogram_subtracted = spectrogram_raw.transpose() - self.mean_spectrum
        spectrogram_subtracted = spectrogram_subtracted.transpose()

        self.pub.publish(self.bridge.cv2_to_imgmsg(spectrogram_subtracted, msg.encoding))


if __name__ == '__main__':
    rospy.init_node('noise_subtractor')
    n = NoiseSubtractor()
    rospy.spin()
