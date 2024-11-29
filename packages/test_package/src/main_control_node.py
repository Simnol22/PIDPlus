#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Float32
from duckietown_msgs.msg import WheelsCmdStamped

from pid_controller import PIDController
import numpy as np
import time

class MaintControlNode(DTROS):
    def __init__(self, node_name):
        super(MaintControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        
        # Reads data from the model node
        # Runs the PID controller and sends the output to the wheel control node

        vehicle_name = os.environ['VEHICLE_NAME']
        wheels_topic = f"/{vehicle_name}/wheels_driver_node/wheels_cmd"
        
        self.radius = 0.0318  # meters, default value of wheel radius
        self.baseline = 0.1  # meters, default value of baseline
        self.gain = 0.7
        self.k = 27
        self.trim = 0

        self.limit = 1
        self.controller = PIDController(kp=13, ki=0, kd=10)
        self.last_time = time.time()
        self.sub = rospy.Subscriber('prediction', Float32, self.callback)
        self.last_prediction = 0

        self._publisher = rospy.Publisher(wheels_topic, WheelsCmdStamped, queue_size=1)
        

    def callback(self, data):
        dt = time.time() - self.last_time
        self.last_time = time.time()
        rospy.loginfo("Predicted : '%s', dt : %s", data.data, str(dt))
        prediction = data.data*0.80 + self.last_prediction*0.20      
        action = self.controller.get_action(prediction,dt)
        self.go(action)
        self.last_prediction = prediction


    def run(self):
        # Determine at what frequence we want to update the PID, 
        # and calculate delat_time
        dt = 50/1000
        while not rospy.is_shutdown():
            start_time = time.time()
            action = self.controller.get_action(self.last_prediction,dt)
            self.go(action)
            time.sleep(50/1000)
            dt = (time.time() - start_time)

        #Call the PID controller 
      
    def go(self, action):
        #Sends voltage to wheels from action

        vel, angle = action

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * self.baseline) / self.radius
        omega_l = (vel - 0.5 * angle * self.baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        message = WheelsCmdStamped(vel_left=vels[0], vel_right=vels[1])
        #print("left vel : ", vels[0])
        #print("right vel : ", vels[1])
        self._publisher.publish(message)


if __name__ == '__main__':
    # create the node
    node = MaintControlNode(node_name='main_control_node')

    # keep the process from terminating
    rospy.spin()