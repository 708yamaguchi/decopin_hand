#!/bin/sh

# Need to use https://github.com/jsk-ros-pkg/jsk_model_tools/pull/225

# Usage
# roscd decopin_hand/model
# ./create_eus_model.sh

rosrun collada_urdf_jsk_patch urdf_to_collada decopin_hand.urdf decopin_hand.dae;
rosrun euscollada collada2eus decopin_hand.dae decopin_hand.yaml ../euslisp/decopin_hand.l;
