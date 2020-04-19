#!/usr/bin/env python

import cv_bridge
from topic_tools import LazyTransport
import numpy as np
import rospy
from sensor_msgs.msg import Image


class NormalizeGrayImage(LazyTransport):
    """
    This class is to convert input gray image to 8FC1 gray image.
    At this time, each pixel is normalized between 0 ~ 255.
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.bridge = cv_bridge.CvBridge()
        self.pub = self.advertise('~output', Image, queue_size=1)
        self.subscribe()

    def subscribe(self):
        self.sub = rospy.Subscriber('~input', Image, self._cb,
                                    queue_size=1, buff_size=2**24)

    def unsubscribe(self):
        self.sub.unregister()

    def _cb(self, imgmsg):
        mono_32fc1 = self.bridge.imgmsg_to_cv2(
            imgmsg, desired_encoding='passthrough')
        _max = mono_32fc1.max()
        _min = mono_32fc1.min()
        mono_32fc1 = (mono_32fc1 - _min) / (_max - _min) * 255.0
        mono_8uc1 = mono_32fc1.astype(np.uint8)
        pubmsg = self.bridge.cv2_to_imgmsg(mono_8uc1, encoding='mono8')
        pubmsg.header = imgmsg.header
        self.pub.publish(pubmsg)


if __name__ == '__main__':
    rospy.init_node('normalize_gray_image')
    sgi = NormalizeGrayImage()
    rospy.spin()
