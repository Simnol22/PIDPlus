#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun pidplus main_control_node.py

# wait for app to end
dt-launchfile-join