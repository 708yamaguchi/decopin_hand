decopin_hand
============

# Usage

1. Create euslisp model.

Before doing this, you must add mimic joint. Please build `jsk_model_tool` including [this pull request](https://github.com/jsk-ros-pkg/jsk_model_tools/pull/225).

```bash
roscd decopin_hand/model
./create_eus_model.sh
../euslisp/decopin_hand_view.l  # for visualization
```

1. Find dynamixels from USB port.

```bash
rosrun decopin_hand find_dynamixel /dev/ttyUSB0
```

1. Start dynamixel controllers. You can change config of dynamixels at `config/yamaguchi_dynamixel
.yaml`

```bash
roslaunch decopin_hand dynamixel_workbench_controllers.launch
```

1. Move dynamixels via roseus

This version do not use euslisp modle and robot-interface.
```bash
roseus euslisp/decopin-interface.l
(decopin-init)
(send *ri* :angle-vector (send *robot* :reset-pose))
```
