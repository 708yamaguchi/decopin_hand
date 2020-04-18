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
```bash
roseus euslisp/decopin_hand_interface.l
(decopin-hand-init)
(send *ri* :angle-vector (send *robot* :reset-pose))
```

```bash
## This version do not use euslisp modle and robot-interface.
roseus euslisp/decopin-interface.l
(decopin-init)
(send *ri* :angle-vector (send *robot* :reset-pose))
```

## Vibration recognition
To all launch files, `use_rosbag` and `filename` arguments can be passed to use rosbag. By default, rosbag is paused at first. Press 'Space' key on terminal to start playing erosbag.

1. Save noise to `train_data/noise.npy`
```
roslaunch decopin_hand save_noise.launch
```

2. Save action spectrograms to `train_data/original_spectrogram/(target_class)`. The newly saveed spectrograms are appended to existing spectrograms.
```
# For action spectrograms
roslaunch decopin_hand save_action.launch target_class:=(target_class) save_when_action:=true
# For non action spectrograms
roslaunch decopin_hand save_action.launch target_class:=no_action save_when_action:=false
```
NOTE
  - `anormal_threshold` argument can be passed. You should set proper threshold, which is effected by size of spectrogram, fft\_exec\_rate ... etc. The higher the threshold is, the harder the spectrograms are saved. You can check whether the threshold is proper by viewing saved spectrograms.
  - Before starting to detect action, some waiting time is required. This is preparation time to calculate mahalanobis distance.

3. Create dateaset for chainer from saved spectrograms. `--number 100` means to use maximum 100 images for each class in dataset.
```
rosrun decopin_hand create_dataset.py --number 100
```

4. Visualize dataset. use `train` for train dataset, `test` for test dataset
```
rosrun decopin_hand visualize_dataset.py train # train/test
```

5. Train with dataset and pretrained weights. First time you run this script, pretrained weights of VGG16 is downloaded to `scripts/VGG_ILSVRC_16_layers.npz`.
```
rosrun decopin_hand train.py
```

6. Classify actions online.
```
roslaunch decopin_hand classify_action.launch
```

## Requirements
For python 2.x, following packages work:
- imageio==2.6.0 (require for imgaug)
- imgaug==0.4.0
- chainer==6.7.0
- cupy-cuda101==6.7.0
