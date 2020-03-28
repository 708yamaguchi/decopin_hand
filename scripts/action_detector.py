#!/usr/bin/env python

import numpy as np
from os import makedirs
from os import path as osp
from sklearn.covariance import MinCovDet

from cv_bridge import CvBridge
import rospkg
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class ActionDetector(object):
    """
    Publish whether the robot is in action or not to rostopic, by MT method.

    NOTE
    Reaction speed for action detection is a bit late
    because spectrum is mean of spectrogram, not right edge of spectrogram
    """

    def __init__(self):
        # Config for loading no action spectrum (noise data)
        rospack = rospkg.RosPack()
        self.train_dir = osp.join(rospack.get_path(
            'decopin_hand'), 'train_data')
        if not osp.exists(self.train_dir):
            makedirs(self.train_dir)
        self.noise_data_path = osp.join(self.train_dir, 'noise.npy')
        if not osp.exists(self.noise_data_path):
            rospy.logerr('{} is not found. Exit.'.format(self.noise_data_path))
            exit()
        no_action_data = np.load(self.noise_data_path)
        # extract about 100 data from no_action_data
        divide = max(1, len(no_action_data) / 100)
        no_action_data = no_action_data[::divide]
        # Detect in action or not by mahalanobis distance
        self.anormal_threshold = rospy.get_param('~anormal_threshold')
        self.mcd = MinCovDet()
        self.mcd.fit(no_action_data)
        rospy.loginfo('Calc covariance matrix for Mahalanobis distance')

        # ROS
        self.bridge = CvBridge()
        self.pub = rospy.Publisher('~in_action', Bool, queue_size=1)
        self.sub = rospy.Subscriber('~raw_spectrogram', Image, self.cb)

    def cb(self, msg):
        """
        Main process of NoiseSaver class
        Publish whether the robot is in action or not
        """

        # spectrogram.shape is (height, width) = (spectrum, time)
        spectrogram = self.bridge.imgmsg_to_cv2(msg)
        self.current_spectrum = np.average(spectrogram, axis=1)
        # Check whether current spectrogram is in action or not
        spectrum = self.current_spectrum[None]
        dist = self.mcd.mahalanobis(spectrum)
        rospy.loginfo(dist)
        if dist < self.anormal_threshold:
            self.in_action = False
            rospy.loginfo('No action')
        else:
            self.in_action = True
            rospy.loginfo('######## In action ########')
        pub_msg = Bool(data=self.in_action)
        self.pub.publish(pub_msg)


if __name__ == '__main__':
    rospy.init_node('action_detector')
    a = ActionDetector()
    rospy.spin()
