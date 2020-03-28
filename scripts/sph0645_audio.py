#!/usr/bin/env python
# -*- coding: utf-8 -*-

# based on https://github.com/jsk-ros-pkg/jsk_3rdparty/blob/master/respeaker_ros/scripts/respeaker_node.py
# To use SPH0645LM4H on Raspberry Pi
# datasheet: https://cdn-shop.adafruit.com/product-files/3421/i2S+Datasheet.PDF

from contextlib import contextmanager
import numpy as np
import os
import pyaudio
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
        self.rate = int(rospy.get_param('~rate', '44100'))
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

        self.channel = min(self.channels - 1, max(0, self.channel))
        self.stream = self.pyaudio.open(
            input=True, start=False, output=False,
            format=pyaudio.paInt32,
            channels=self.channels,
            rate=self.rate,
            frames_per_buffer=1024,
            stream_callback=self.stream_callback,
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

    def stream_callback(self, in_data, frame_count, time_info, status):
        # split channel
        data = np.fromstring(in_data, np.int32)
        data = data >> 14; # This 18bit integer is raw data from microphone
        data = (data >> 2).astype(np.int16)
        chunk_per_channel = len(data) / self.channels
        data = np.reshape(data, (chunk_per_channel, self.channels))
        chan_data = data[:, self.channel]
        # invoke callback
        self.pub_audio.publish(AudioData(data=chan_data.tostring()))
        return None, pyaudio.paContinue

    def start(self):
        if self.stream.is_stopped():
            self.stream.start_stream()

    def stop(self):
        if self.stream.is_active():
            self.stream.stop_stream()


if __name__ == '__main__':
    rospy.init_node('audio_capture_microphone', anonymous=True)
    s = SPH0645Audio()
    rospy.spin()