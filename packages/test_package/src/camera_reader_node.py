#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage

import cv2
from cv_bridge import CvBridge
import numpy as np
from PIL import Image
import torch

class CameraReaderNode(DTROS):

    def __init__(self, node_name):

        # initialize the DTROS parent class
        super(CameraReaderNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        # static parameters
        print("THIS IS A TEST")
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        # bridge between OpenCV and ROS
        self._bridge = CvBridge()
        # create window
        self._window = "camera-reader"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        self.ii = 0
        os.mkdir("images")

    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        
       # image_name = f"images/image_{self.ii}.jpg"

        #cv2.imwrite(image_name,image)
        #print(image_name)
        #image = apply_preprocessing(image)
        # display frame
        print("THIS IS A TEST")
        cv2.imshow(self._window, image)
        self.ii += 1
    
def apply_preprocessing(image):
    """
    Apply preprocessing transformations to the input image.

    Parameters:
    - image: PIL Image object.
    """
    image_array = np.array(image)
    channels = [image_array[:, :, i] for i in range(image_array.shape[2])]
    h, w, _ = image_array.shape
    
    imghsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    mask_ground = np.ones(img.shape, dtype=np.uint8)  # Start with a mask of ones (white)


    one_third_height = h // 3
    mask_ground[:one_third_height, :] = 0  # Mask the top 1/3 of the image
    
    #gaussian filter
    sigma = 4.5
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
    
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    threshold = 50


    white_lower_hsv = np.array([0,(0*255)/100,(60*255)/100]) # [0,0,50] - [230,100,255]
    white_upper_hsv = np.array([150,(40*255)/100,(100*255)/100])   # CHANGE ME

    yellow_lower_hsv = np.array([(30*179)/360, (30*255)/100, (30*255)/100])        # CHANGE ME
    yellow_upper_hsv = np.array([(90*179)/360, (110*255)/100, (100*255)/100])  # CHANGE ME

    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)


    mask_mag = (Gmag > threshold)

    # np.savetxt("mask.txt", mask_white, fmt='%d', delimiter=',')
    # exit()

    final_mask = mask_ground * mask_mag * 255 
    mask_white = mask_ground * mask_white
    mask_yellow = mask_ground * mask_yellow
    # Convert the NumPy array back to a PIL image

    channels[0] =  np.zeros_like(channels[0])# final_mask
    channels[1] =  np.zeros_like(channels[1]) #mask_white
    channels[2] =  mask_yellow
    
    filtered_image = np.stack(channels, axis=-1)
    #filtered_image = Image.fromarray(filtered_image)
    return  filtered_image


if __name__ == '__main__':
    # create the node
    node = CameraReaderNode(node_name='camera_reader_node')
    # keep spinning
    rospy.spin()