#!/usr/bin/env python

from cv_bridge import CvBridge
from decopin_hand.msg import InAction
from sensor_msgs.msg import Image
import rospy
from topic_tools import LazyTransport


class ActionDetectorHistogram(LazyTransport):
    """
    Detect whether the robot is in action or not by color histogram.

    Robot is detected as 'in_action' when
    the number of pixels which have less value than 'pixel_value_threshold'
    is more than (the number of total pixels) * 'pixel_ratio_threshold'
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        # ROS
        self.threshold = rospy.get_param('~power_per_pixel_threshold', 0)
        self.pub = self.advertise('~in_action', InAction, queue_size=1)
        self.cv_bridge = CvBridge()

    def subscribe(self):
        self.sub = rospy.Subscriber('~input', Image, self._cb)

    def unsubscribe(self):
        self.sub.unregister()

    def _cb(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg)
        pub_msg = InAction()
        pub_msg.header = msg.header
        power_per_pixel = img.sum() / img.size
        rospy.logdebug('power_per_pixel: {}, threshold: {}'.format(
            power_per_pixel, self.threshold))
        if power_per_pixel > self.threshold:
            rospy.logdebug('### In action ###')
            pub_msg.in_action = True
        else:
            rospy.logdebug('No action')
            pub_msg.in_action = False
        self.pub.publish(pub_msg)


if __name__ == '__main__':
    rospy.init_node('action_detector_histogram')
    a = ActionDetectorHistogram()
    rospy.spin()
