#!/usr/bin/env python

# Copied from https://github.com/jsk-ros-pkg/jsk_recognition/blob/master/jsk_perception/node_scripts/vgg16_object_recognition.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import chainer
from chainer import cuda
import chainer.serializers as S
from chainer import Variable
from distutils.version import LooseVersion
import numpy as np
import skimage.transform

import cv_bridge
from jsk_recognition_msgs.msg import ClassificationResult
from jsk_recognition_utils.chainermodels import VGG16
from jsk_recognition_utils.chainermodels import VGG16BatchNormalization
from jsk_topic_tools import ConnectionBasedTransport
import rospy
from sensor_msgs.msg import Image

from train import PreprocessedDataset


class ActionClassifier(ConnectionBasedTransport):

    def __init__(self):
        super(self.__class__, self).__init__()
        self.insize = 224
        self.gpu = rospy.get_param('~gpu', -1)
        self.dataset = PreprocessedDataset()
        self.target_names = self.dataset.target_classes
        self.model_name = rospy.get_param('~model_name')
        if self.model_name == 'vgg16':
            self.model = VGG16(n_class=len(self.target_names))
        elif self.model_name == 'vgg16_batch_normalization':
            self.model = VGG16BatchNormalization(
                n_class=len(self.target_names))
        else:
            rospy.logerr('Unsupported ~model_name: {0}'
                         .format(self.model_name))
        model_file = rospy.get_param('~model_file')
        S.load_npz(model_file, self.model)
        if self.gpu != -1:
            self.model.to_gpu(self.gpu)
        self.pub = self.advertise('~output', ClassificationResult,
                                  queue_size=1)
        self.pub_input = self.advertise(
            '~debug/net_input', Image, queue_size=1)

    def subscribe(self):
        sub = rospy.Subscriber(
            '~input', Image, self._recognize, callback_args=None,
            queue_size=1, buff_size=2**24)
        self.subs = [sub]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()

    def _recognize(self, imgmsg):
        bridge = cv_bridge.CvBridge()
        bgr = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='bgr8')
        bgr = skimage.transform.resize(
            bgr, (self.insize, self.insize), preserve_range=True)
        input_msg = bridge.cv2_to_imgmsg(bgr.astype(np.uint8), encoding='bgr8')
        input_msg.header = imgmsg.header
        self.pub_input.publish(input_msg)

        # (Height, Width, Channel) -> (Channel, Height, Width)
        rgb = bgr.transpose((2, 0, 1))[::-1, :, :]
        rgb = self.dataset.process_image(rgb)
        x_data = np.array([rgb], dtype=np.float32)
        if self.gpu != -1:
            x_data = cuda.to_gpu(x_data, device=self.gpu)
        if LooseVersion(chainer.__version__) < LooseVersion('2.0.0'):
            x = Variable(x_data, volatile=True)
            self.model.train = False
            self.model(x)
        else:
            with chainer.using_config('train', False), \
                    chainer.no_backprop_mode():
                x = Variable(x_data)
                self.model(x)

        proba = cuda.to_cpu(self.model.pred.data)[0]
        label = np.argmax(proba)
        label_name = self.target_names[label]
        label_proba = proba[label]
        cls_msg = ClassificationResult(
            header=imgmsg.header,
            labels=[label],
            label_names=[label_name],
            label_proba=[label_proba],
            probabilities=proba,
            classifier=self.model_name,
            target_names=self.target_names,
        )
        self.pub.publish(cls_msg)


if __name__ == '__main__':
    rospy.init_node('action_classifier')
    app = ActionClassifier()
    rospy.spin()
