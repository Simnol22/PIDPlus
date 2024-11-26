#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# launch subscriber
rosrun test_package subscriber_node.py &
rosrun test_package publisher_node.py &
wait
# wait for app to end
dt-launchfile-join