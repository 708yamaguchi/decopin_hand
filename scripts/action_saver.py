#!/usr/bin/env python

from os import makedirs, listdir
from os import path as osp
from PIL import Image as Image_

from cv_bridge import CvBridge
from decopin_hand.msg import InAction
import message_filters
import rospkg
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class ActionSaver(object):
    """
    Collect spectrogram with action class, only when the robot is in action.
    if save_when_action is False, you can save spectrograms during no action.
    """

    def __init__(self):
        # Config for saving spectrogram
        target_class = rospy.get_param('~target_class')
        rospack = rospkg.RosPack()
        self.train_dir = osp.join(rospack.get_path(
            'decopin_hand'), 'train_data')
        if not osp.exists(self.train_dir):
            makedirs(self.train_dir)
        self.image_save_dir = osp.join(
            self.train_dir, 'original_spectrogram', target_class)
        if not osp.exists(self.image_save_dir):
            makedirs(self.image_save_dir)
        # ROS
        self.bridge = CvBridge()
        self.save_data_rate = rospy.get_param('~save_data_rate')
        self.save_when_action = rospy.get_param('~save_when_action')
        self.in_action = False
        self.spectrogram_msg = None
        img_sub = message_filters.Subscriber('~input', Image)
        in_action_sub = message_filters.Subscriber('~in_action', InAction)
        ts = message_filters.TimeSynchronizer([img_sub, in_action_sub], 1)
        ts.registerCallback(self._cb)
        rospy.Timer(rospy.Duration(1. / self.save_data_rate), self.timer_cb)

    def _cb(self, img, in_action):
        self.spectrogram_msg = img
        self.in_action = in_action.in_action

    def timer_cb(self, timer):
        """
        Main process of NoiseSaver class
        Save spectrogram data at self.save_data_rate
        """

        if self.spectrogram_msg is None:
            return
        if self.save_when_action is True and self.in_action is False:
            pass
        else:
            file_num = len(
                listdir(self.image_save_dir)) + 1  # start from 00001.npy
            file_name = osp.join(
                self.image_save_dir, '{0:05d}.png'.format(file_num))
            mono_spectrogram = self.bridge.imgmsg_to_cv2(self.spectrogram_msg)
            Image_.fromarray(mono_spectrogram).save(file_name)
            rospy.loginfo('save spectrogram: ' + file_name)


if __name__ == '__main__':
    rospy.init_node('action_saver')
    a = ActionSaver()
    rospy.spin()
