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

3. Find dynamixels from USB port.
```bash
rosrun decopin_hand find_dynamixel /dev/ttyUSB0
```

4. Start dynamixel controllers. You can change config of dynamixels at `config/yamaguchi_dynamixel
.yaml`

```bash
roslaunch decopin_hand dynamixel_workbench_controllers.launch
```

5. Move dynamixels via roseus
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

## Vibration recognition
To all launch files, `use_rosbag` and `filename` arguments can be passed to use rosbag. By default, rosbag is paused at first. Press 'Space' key on terminal to start playing rosbag.

0. Publish `/audio` topic in raspberry pi.
```bash
rosrun decopin_hand sph0645_audio.py
```
Then, You can see the spectrograms calculated from `/audio` by `roslaunch decopin_hand audio_to_spectrogram.launch`

1. Save noise to `train_data/noise.npy`. By subtracting noise, spectrograms become clear.
```bash
roslaunch decopin_hand save_noise.launch
```

2. Record rosbag to save aciton spectrograms because we can get train data repeatedly from rosbag.\
   You can record rosbag by the following commands:
```bash
# To record rosbag with /audio topic
roslaunch decopin_hand audio_to_spectrogram.launch
roslaunch decopin_hand record_audio_rosbag.launch filename:=$HOME/.ros/rosbag/hoge.bag
```

3. Save action spectrograms to `train_data/original_spectrogram/(target_class)`. The newly saveed spectrograms are appended to existing spectrograms.\
   By using `use_rosbag:=true` and `filename:=xxx`, you can save action spectrograms from rosbag. When using rosbag, **DO NOT** publish `/audio` topic from real microphone.
```bash
# For action spectrograms
roslaunch decopin_hand save_action.launch target_class:=(target_class) save_when_action:=true # use_rosbag:=true filename:=$HOME/.ros/rosbag/hoge.bag
# For non action spectrograms
roslaunch decopin_hand save_action.launch target_class:=no_action save_when_action:=false # use_rosbag:=true filename:=$HOME/.ros/rosbag/hoge.bag
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

## Requirements
For python 2.x, following packages work:
- imageio==2.6.0 (require for imgaug)
- imgaug==0.4.0
- chainer==6.7.0
- cupy-cuda101==6.7.0
