
#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

from pid_controller import PIDController

class MaintControlNode(DTROS):
    def __init__(self, node_name):
        super(MaintControlNode, self).__init__(node_name=node_name, node_type=NodeType.LOCALIZATION)
        
        # Reads data from the model node
        # Runs the PID controller and sends the output to the wheel control node
        self.controller = PIDController()

        self.sub = rospy.Subscriber('prediction', Float32, self.callback)
        self.last_prediction = 0

    def callback(self, data):
        self.last_prediction = data.data
        rospy.loginfo("Predicted : '%s'", data.data)

    def run(self):
        #Call the PID controller 
        self.controller.compute(self.last_prediction)
        pass

if __name__ == '__main__':
    # create the node
    node = MaintControlNode(node_name='wheel_control_node')
    # run node
    node.run()
    # keep the process from terminating
    rospy.spin()