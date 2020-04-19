#!/usr/bin/env python

from os import makedirs, listdir
from os import path as osp
from PIL import Image as Image_

from cv_bridge import CvBridge
from decopin_hand.msg import InAction
import message_filters
import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import Image


class ActionSaver(object):
    """
    Collect spectrogram with action class, only when the robot is in action.
    if save_when_action is False, you can save spectrograms during no action.
    """

    def __init__(self):
        # Config for saving spectrogram
        target_class = rospy.get_param('~target_class')
        self.save_raw_spectrogram = rospy.get_param('~save_raw_spectrogram')
        rospack = rospkg.RosPack()
        self.train_dir = osp.join(rospack.get_path(
            'decopin_hand'), 'train_data')
        if not osp.exists(self.train_dir):
            makedirs(self.train_dir)
        self.image_save_dir = osp.join(
            self.train_dir, 'original_spectrogram', target_class)
        if not osp.exists(self.image_save_dir):
            makedirs(self.image_save_dir)
        self.raw_image_save_dir = osp.join(self.image_save_dir, 'raw')
        if not osp.exists(self.raw_image_save_dir):
            makedirs(self.raw_image_save_dir)
        # ROS
        self.bridge = CvBridge()
        self.save_data_rate = rospy.get_param('~save_data_rate')
        self.save_when_action = rospy.get_param('~save_when_action')
        self.in_action = False
        self.spectrogram_msg = None
        self.spectrogram_raw_msg = None
        in_action_sub = message_filters.Subscriber('~in_action', InAction)
        img_sub = message_filters.Subscriber('~input', Image)
        img_raw_sub = message_filters.Subscriber('~input_raw', Image)
        subs = [in_action_sub, img_sub]
        if self.save_raw_spectrogram:
            subs.append(img_raw_sub)
        ts = message_filters.TimeSynchronizer(subs, 100000)
        ts.registerCallback(self._cb)
        rospy.Timer(rospy.Duration(1. / self.save_data_rate), self.timer_cb)

    def _cb(self, *args):
        in_action = args[0].in_action
        # rospy.logerr('in_action: {}'.format(in_action))
        if self.save_when_action is False:
            in_action = True
        if in_action:
            self.spectrogram_msg = args[1]
            if self.save_raw_spectrogram:
                self.spectrogram_raw_msg = args[2]
        else:
            self.spectrogram_msg = None
            if self.save_raw_spectrogram:
                self.spectrogram_raw_msg = None

    def timer_cb(self, timer):
        """
        Main process of NoiseSaver class
        Save spectrogram data at self.save_data_rate
        """

        if self.spectrogram_msg is None or self.spectrogram_raw_msg is None:
            return
        else:
            file_num = len(
                listdir(self.image_save_dir)) + 1  # start from 00001.npy
            file_name = osp.join(
                self.image_save_dir, '{0:05d}.png'.format(file_num))
            mono_spectrogram = self.bridge.imgmsg_to_cv2(self.spectrogram_msg)
            Image_.fromarray(mono_spectrogram).save(file_name)
            # self.spectrogram_msg = None
            rospy.loginfo('save spectrogram: ' + file_name)
            if self.save_raw_spectrogram:
                file_name_raw = osp.join(
                    self.raw_image_save_dir, '{0:05d}_raw.png'.format(file_num))
                mono_spectrogram_raw = self.bridge.imgmsg_to_cv2(
                    self.spectrogram_raw_msg, desired_encoding='32FC1')
                _max = mono_spectrogram_raw.max()
                _min = mono_spectrogram_raw.min()
                mono_spectrogram_raw = (mono_spectrogram_raw - _min) / (_max - _min) * 255.0
                mono_spectrogram_raw = mono_spectrogram_raw.astype(np.uint8)
                Image_.fromarray(mono_spectrogram_raw).save(file_name_raw)
                # self.spectrogram_raw_msg = None
                rospy.loginfo('save spectrogram: ' + file_name_raw)


if __name__ == '__main__':
    rospy.init_node('action_saver')
    a = ActionSaver()
    rospy.spin()
