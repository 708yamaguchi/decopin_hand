#!/usr/bin/env python

import cv_bridge
import numpy as np
import os.path as osp
import rospkg
import rospy
from sensor_msgs.msg import Image
from topic_tools import LazyTransport
from process_gray_image import noise_subtract, smooth_gray_image, normalize_gray_image


class PreprocessGrayImage(LazyTransport):
    """
    This class is to preprocess gray spectrogram for classifying.
    1. Noise subtraction by spectral subtraction method
    2. Smooth spectrogram
    3. Normalize spectrogram (32FC1 -> 8UC1, make each pixel value 0 ~ 255)
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        # Noise subtraction
        rospack = rospkg.RosPack()
        self.train_dir = osp.join(rospack.get_path(
            'decopin_hand'), 'train_data')
        self.noise_data_path = osp.join(self.train_dir, 'noise.npy')
        if osp.exists(self.noise_data_path):
            noise_data = np.load(self.noise_data_path)
            self.mean_spectrum = np.mean(noise_data, axis=0)
        else:
            rospy.logwarn('{} is not found.'.format(self.noise_data_path))
            self.mean_spectrum = 0
        # Publisher and Subscriber
        self.bridge = cv_bridge.CvBridge()
        self.pub = self.advertise('~output', Image, queue_size=1)
        self.subscribe()

    def subscribe(self):
        self.sub = rospy.Subscriber('~input', Image, self._process,
                                    queue_size=1, buff_size=2**24)

    def unsubscribe(self):
        self.sub.unregister()

    def _process(self, imgmsg):
        raw_img = self.bridge.imgmsg_to_cv2(imgmsg)
        # Noise subtract
        subtracted_img = noise_subtract(raw_img, self.mean_spectrum)
        # Blur
        blured_img = smooth_gray_image(subtracted_img)
        # Normalize
        blured_8uc1_img = normalize_gray_image(blured_img)
        # Publish
        pubmsg = self.bridge.cv2_to_imgmsg(blured_8uc1_img)
        pubmsg.header = imgmsg.header
        self.pub.publish(pubmsg)


if __name__ == '__main__':
    rospy.init_node('preprocess_gray_image')
    pgi = PreprocessGrayImage()
    rospy.spin()
