#!/usr/bin/env python
# -*- coding: utf-8 -*-

# based on https://github.com/jsk-ros-pkg/jsk_3rdparty/blob/master/respeaker_ros/scripts/respeaker_node.py
# To use SPH0645LM4H on Raspberry Pi
# datasheet: https://cdn-shop.adafruit.com/product-files/3421/i2S+Datasheet.PDF

from contextlib import contextmanager
import numpy as np
import os
import pyaudio
import signal
import sys

import rospy
from audio_common_msgs.msg import AudioData

# suppress error messages from ALSA
# https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time
# https://stackoverflow.com/questions/36956083/how-can-the-terminal-output-of-executables-run-by-python-functions-be-silenced-i
@contextmanager
def ignore_stderr(enable=True):
    if enable:
        devnull = None
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            stderr = os.dup(2)
            sys.stderr.flush()
            os.dup2(devnull, 2)
            try:
                yield
            finally:
                os.dup2(stderr, 2)
                os.close(stderr)
        finally:
            if devnull is not None:
                os.close(devnull)
    else:
        yield


class SPH0645Audio(object):
    def __init__(self, channel=0, suppress_error=True):
        try:
            self.stop()
        except:
            pass
        with ignore_stderr(enable=suppress_error):
            self.pyaudio = pyaudio.PyAudio()
        self.name = rospy.get_param('~name', 'snd_rpi_simple_card')
        self.rate = int(rospy.get_param('~rate', 44100))
        self.frame_per_buffer = int(rospy.get_param('~frame_per_buffer', 1024))
        # bit depth is assumed as 32 (paInt32). This is based on SPH0645LM4H.
        self.channels = None
        self.channel = channel
        self.device_index = None
        self.pub_audio = rospy.Publisher("audio", AudioData, queue_size=10)

        # find device
        count = self.pyaudio.get_device_count()
        rospy.logdebug("%d audio devices found" % count)
        for i in range(count):
            info = self.pyaudio.get_device_info_by_index(i)
            name = info["name"].encode("utf-8")
            chan = info["maxInputChannels"]
            rospy.logdebug(" - %d: %s" % (i, name))
            if name.lower().find(self.name) >= 0:
                self.channels = chan
                self.device_index = i
                rospy.loginfo("Found %d: %s (channels: %d)" % (i, name, chan))
                break
        if self.device_index is None:
            rospy.logwarn("Failed to find microphone by name. Using default input")
            info = self.pyaudio.get_default_input_device_info()
            self.channels = info["maxInputChannels"]
            self.device_index = info["index"]

        # self.channels is the number of channels of the microphone
        # self.channel is the target channel to get single channel data
        self.channel = min(self.channels - 1, max(0, self.channel))
        # Do not use callback if you do not want sensor noise
        # Noise is, for example, 0b11111111 data.
        self.stream = self.pyaudio.open(
            input=True, output=False,
            format=pyaudio.paInt32,
            channels=self.channels,
            rate=self.rate,
            frames_per_buffer=self.frame_per_buffer,
            input_device_index=self.device_index,
        )
        self.start()

    def __del__(self):
        self.stop()
        try:
            self.stream.close()
        except:
            pass
        finally:
            self.stream = None
        try:
            self.pyaudio.terminate()
        except:
            pass

    def start(self):
        if self.stream.is_stopped():
            self.stream.start_stream()

    def stop(self):
        if self.stream.is_active():
            self.stream.stop_stream()

    def kill(self, signum, frame):
        rospy.loginfo('Ctrl-c is pressed. Exit')
        self.stop()

if __name__ == '__main__':
    rospy.init_node('audio_capture_microphone')
    s = SPH0645Audio()
    r = rospy.Rate(float(s.rate) / s.frame_per_buffer)
    signal.signal(signal.SIGINT, s.kill)
    while s.stream.is_active():
        # Input data to 16 bit
        in_data = np.frombuffer(s.stream.read(1024), np.int32)
        in_data = in_data >> 14  # This 18bit integer is raw data from microphone
        int16_data = (in_data >> 2).astype(np.int16)
        # Retreive 1 channel data
        chunk_per_channel = len(int16_data) / s.channels
        channel_data = int16_data[s.channel::s.channels]
        # Publish audio topic
        s.pub_audio.publish(AudioData(data=channel_data.tostring()))
        r.sleep()
