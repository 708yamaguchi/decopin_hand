decopin_hand
============

## Control hand
1. Build `jsk_model_tools` including [this pull request](https://github.com/jsk-ros-pkg/jsk_model_tools/pull/225) to add mimic joint.

2. Create euslisp model.
```bash
roscd decopin_hand/model
./create_eus_model.sh
../euslisp/decopin_hand_view.l  # for visualization
```

3. Add write permission to `/dev/ttyUSB*`.
Write following message to `/etc/udev/rules.d/99-dynamixel-workbench-cdc.rules`. (Please create file if missing)
```
KERNEL=="ttyUSB*", DRIVERS=="ftdi_sio", MODE="0666", ATTR{device/latency_timer}="1"
````

4. Find dynamixels from USB port.
```bash
rosrun decopin_hand find_dynamixel /dev/ttyUSB0
```

5. Start dynamixel controllers. You can change config of dynamixels at `config/yamaguchi_dynamixel
.yaml`

```bash
roslaunch decopin_hand dynamixel_workbench_controllers.launch
```

6. Move dynamixels via roseus
The robot class is inherited from robot-interface.
```lisp
roseus euslisp/decopin_hand_interface.l
(decopin-hand-init)
(send *ri* :angle-vector (send *robot* :reset-pose))
```

```lisp
## This version do not use euslisp modle and robot-interface.
roseus euslisp/decopin-interface.l
(decopin-init)
(send *ri* :angle-vector (send *robot* :reset-pose))
```

7. Move dynamixels via nanoKONTROL2
After launching this file, you can control dynamixels from sliders on the controller. The control state is managed by smach state machine. You can see the state on smach viewer.
```bash
roslaunch decopin_hand decopin_hand_kontrol.launch
```

## Vibration recognition
To all launch files, `use_rosbag` and `filename` arguments can be passed to use rosbag. By default, rosbag is paused at first. Press 'Space' key on terminal to start playing rosbag.

0. Publish `/audio` topic in raspberry pi.
```bash
# audio only
rosrun decopin_hand sph0645_audio.py
# with motor control
roslaunch decopin_hand mic_and_motor.launch
```
Then, You can see the spectrograms calculated from `/audio` by `roslaunch decopin_hand audio_to_spectrogram.launch`

1. Save noise to `train_data/noise.npy`. By subtracting noise, spectrograms become clear. During this script, you must not give vibration to the sensor. You should update noise data everytime before vibration recognition because environmental vibration noise differs everytime.
```bash
roslaunch decopin_hand save_noise.launch
```

2. Record rosbag to save aciton spectrograms because we can get train data repeatedly from rosbag.\
   You can record rosbag by the following commands:
```bash
# To record rosbag with /audio topic
roslaunch decopin_hand audio_to_spectrogram.launch gui:=true
roslaunch decopin_hand record_audio_rosbag.launch filename:=$HOME/.ros/rosbag/hoge.bag
```

3. Save action spectrograms to `train_data/original_spectrogram/(target_class)`. The newly saveed spectrograms are appended to existing spectrograms.\
   NOTE
   - By using `use_rosbag:=true` and `filename:=xxx`, you can save action spectrograms from rosbag. When using rosbag, **DO NOT** publish `/audio` topic from real microphone.
   - You can change threshold of action saving by `threshold:=xxx`. The smaller the value is, the more easily action is saved.
```bash
rossetlocal # if you want to use rosbag
# For action spectrograms
roslaunch decopin_hand save_action.launch target_class:=(target_class) save_when_action:=true use_rosbag:=true threshold:=0.5 save_data_rate:=5 filename:=$HOME/.ros/rosbag/hoge.bag
# For non action spectrograms
roslaunch decopin_hand save_action.launch target_class:=no_action save_when_action:=false use_rosbag:=true filename:=$HOME/.ros/rosbag/hoge.bag
```

4. Create dateaset for chainer from saved spectrograms. `--number 100` means to use maximum 100 images for each class in dataset.
```bash
rosrun decopin_hand create_dataset.py --number 100
```

5. Visualize dataset. use `train` for train dataset, `test` for test dataset
```bash
rosrun decopin_hand visualize_dataset.py train # train/test
```

6. Train with dataset. Default model is `NIN`. If you use `vgg16`, pretrained weights of VGG16 is downloaded to `scripts/VGG_ILSVRC_16_layers.npz` at the first time you run this script.
```bash
rosrun decopin_hand train.py --epoch 30
```

7. Classify actions online.
```bash
roslaunch decopin_hand classify_action.launch
```

8. Visualize spectrogram processing and save spectrograms at each processing stages. This script uses `original_spectrogram` directory created by `action_saver.py`
```bash
rosrun decopin_hand visualize_spectrogram_process.py -t (target_class)
```
Visualize classification result. You can save file by right-click of the image_view window. Input file path is `$(arg spectrogram_dir)/$(arg input_file_name)` and output file path is `$(arg spectrogram_dir)/$(arg output_file_name)` You can use output file of `visualize_spectrogram_process.py` as input of this script.
```bash
roslaunch decopin_hand classify_action_static.launch spectrogram_dir:=xxx input_file_name:=yyy
```

## Requirements
For python 2.x, following packages work:
- imageio==2.6.0 (require for imgaug)
- imgaug==0.4.0
- chainer==6.7.0
- cupy-cuda101==6.7.0
