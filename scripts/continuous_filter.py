#!/usr/bin/env python

from jsk_recognition_msgs.msg import ClassificationResult
import rospy
from topic_tools import LazyTransport


class ContinuousFilter(LazyTransport):
    """
    Overview:
    If the input class is the same class for duration_thre, and
       dead_time after the last gesture,
    this node outputs the neural network output class as the gesture,
    Else, this node outputs none class.

    Member variables:
    self.temporary_class  : The current class
    self.continuous_class : The class which continues for duration_thre[s]
    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.none_class = 'none\n'
        self.temporary_class = self.none_class
        self.continuous_class = self.none_class
        self.duration_thre = rospy.get_param('~duration_thre', 0.3)
        self.last_temporary_class_time = rospy.Time()
        self.previous_valid_gesture = self.none_class
        self.name_to_label = {}
        self.name_to_label[self.none_class] = 0
        self.pub = self.advertise('~output', ClassificationResult,
                                  queue_size=1)
        self.subscribe()

    def subscribe(self):
        self.sub = rospy.Subscriber(
            '~input', ClassificationResult, self.cb,
            queue_size=1, buff_size=2**24)

    def unsubscribe(self):
        self.sub.unregister()

    def cb(self, msg):
        label_name = msg.label_names[0]
        if label_name not in self.name_to_label:
            self.name_to_label[label_name] = msg.labels[0]
        # Filtering class for gesture recognition
        now_time = rospy.Time.now()
        if label_name != self.temporary_class:
            self.last_temporary_class_time = rospy.Time.now()
        self.temporary_class = label_name
        # Update continous class
        if (now_time - self.last_temporary_class_time).to_sec() > self.duration_thre:
            self.continuous_class = self.temporary_class
        # Publish continous msg
        probabilities = [0.0] * len(msg.probabilities)
        probabilities[0] = 1.0
        pubmsg = ClassificationResult(
            header=msg.header,
            labels=[self.name_to_label[self.continuous_class]],
            label_names=[self.continuous_class],
            label_proba=[1.0],
            probabilities=probabilities,
            classifier=msg.classifier,
            target_names=msg.target_names,
        )
        self.pub.publish(pubmsg)


if __name__ == '__main__':
    rospy.init_node('continuous_filter')
    cf = ContinuousFilter()
    rospy.spin()
