#!/usr/bin/env python

from jsk_recognition_msgs.msg import ClassificationResult
import rospy
from topic_tools import LazyTransport


class GestureFilter(LazyTransport):
    """
    Overview:
    If the input class is the same class for duration_thre, and
       dead_time after the last gesture,
    this node outputs the neural network output class as the gesture,
    Else, this node outputs none class.

    Member variables:
    self.temporary_class  : The current class
    self.continuous_class : The class which continues for duration_thre[s]
    self.gesture_class    : The continuous_class after dead_time[s]

    """

    def __init__(self):
        super(self.__class__, self).__init__()
        self.none_class = 'none\n'
        self.dead_time_class = 'dead_time\n'
        self.temporary_class = self.none_class
        self.continuous_class = self.none_class
        self.gesture_class = self.none_class
        self.duration_thre = rospy.get_param('~duration_thre', 0.4)
        self.last_temporary_class_time = rospy.Time()
        self.dead_time = rospy.get_param('~dead_time', 8)
        self.gesture_started_time = rospy.Time()
        self.previous_valid_gesture = self.none_class
        self.name_to_label = {}
        self.name_to_label[self.none_class] = 0
        self.name_to_label[self.dead_time_class] = 0
        self.pub = self.advertise('~output', ClassificationResult,
                                  queue_size=1)
        self.subscribe()

    def subscribe(self):
        self.sub = rospy.Subscriber(
            '~input', ClassificationResult, self.cb,
            queue_size=1, buff_size=2**24)

    def unsubscribe(self):
        self.sub.unregister()

    def invalid_gesture_check(self, current_gesture):
        """
        This function checks if current_gesture is valid based on
        self.previous_gesture.
        This function changes depending on experiments.
        """
        return current_gesture

    def cb(self, msg):
        label_name = msg.label_names[0]
        if label_name not in self.name_to_label:
            self.name_to_label[label_name] = msg.labels[0]
        # Filtering class for gesture recognition
        now_time = rospy.Time.now()
        if label_name != self.temporary_class:
            # self.gesture_class = self.none_class
            self.last_temporary_class_time = rospy.Time.now()
        self.temporary_class = label_name
        # Update continous class
        if (now_time - self.last_temporary_class_time).to_sec() > self.duration_thre:
            self.continuous_class = self.temporary_class
        # Update gesture class
        if self.continuous_class != self.none_class:
            if (now_time - self.gesture_started_time).to_sec() > self.dead_time:
                self.gesture_class = self.continuous_class
                self.gesture_started_time = now_time
            else:
                if self.continuous_class != self.gesture_class:
                    self.gesture_class = self.dead_time_class
        else:
            if (now_time - self.gesture_started_time).to_sec() > self.dead_time:
                self.gesture_class = self.none_class
            else:
                self.gesture_class = self.dead_time_class
        # Check if the gesture is valid
        self.gesture_class = self.invalid_gesture_check(self.gesture_class)
        if self.gesture_class != self.none_class and\
           self.gesture_class != self.dead_time_class:
            self.previous_valid_gesture = self.gesture_class
        # print('~~~~~~~~~~~')
        # print('gesture_class: {}'.format(self.gesture_class))
        # print('gesture elapsed time: {}'.format((now_time - self.gesture_started_time).to_sec()))

        # Publish continous msg
        probabilities = [0.0] * len(msg.probabilities)
        probabilities[0] = 1.0
        pubmsg = ClassificationResult(
            header=msg.header,
            labels=[self.name_to_label[self.gesture_class]],
            label_names=[self.gesture_class],
            label_proba=[1.0],
            probabilities=probabilities,
            classifier=msg.classifier,
            target_names=msg.target_names,
        )
        self.pub.publish(pubmsg)


if __name__ == '__main__':
    rospy.init_node('gesture_filter')
    cf = GestureFilter()
    rospy.spin()
