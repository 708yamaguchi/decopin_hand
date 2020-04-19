#!/usr/bin/env python

import cv2
import cv_bridge
from topic_tools import LazyTransport
import rospy
from sensor_msgs.msg import Image
from process_gray_image import ProcessGrayImage

class SmoothGrayImage(LazyTransport):
    """
    This class is to smooth gray image. (32FC1 is assumed)
    """

    def __init__(self):
        super(self.__class__, self).__init__()

    def _process(self, imgmsg):
        raw_img = self.bridge.imgmsg_to_cv2(imgmsg)
        smoothed_img = cv2.blur(raw_img, (5, 5))
        pubmsg = self.bridge.cv2_to_imgmsg(smoothed_img)
        pubmsg.header = imgmsg.header
        self.pub.publish(pubmsg)


if __name__ == '__main__':
    rospy.init_node('smooth_gray_image')
    sgi = SmoothGrayImage()
    rospy.spin()
