#!/usr/bin/env python3

import os
import rospy
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

import cv2
from cv_bridge import CvBridge
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from pid_controller import PIDController
from lane_detection_model import LaneDetectionCNN

class ModelNode(DTROS):
    def __init__(self, node_name):
        super(ModelNode, self).__init__(node_name=node_name, node_type=NodeType.VISUALIZATION)
        
        # Reads data from the camera, preprocesses it and sends it through the model
        # Outputs the prediction to the a topic
        self._vehicle_name = os.environ['VEHICLE_NAME']
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._bridge = CvBridge()

        # create window if necessary
        self._window = "model-node"
        cv2.namedWindow(self._window, cv2.WINDOW_AUTOSIZE)
        # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        # create publisher
        self.pub = rospy.Publisher('prediction', Float32, queue_size=10)
        
        self.model_type = "CNN"
        self.model = None
        
        if self.model_type == "CNN":
            self.model = LaneDetectionCNN((3, 224, 224))
            self.load_model("packages/test_package/src/model/latest_cnn.pth")

        elif self.model_type == "RNN":
            ...

        #Transformations to apply to an image before sending it to the model
        self.transformCNN = transforms.Compose([
            transforms.Lambda(self.apply_preprocessing_cnn),
            transforms.Lambda(Image.fromarray),
            transforms.Resize((224, 224)),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor()  # Convert to tensor
        ])

    def load_model(self, model_path):
        # Load the model
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model.eval()

    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        im = image
        # preprocess image for viz
        if self.model_type == "CNN":
            im = self.apply_preprocessing_cnn(im)
        
        cv2.imshow(self._window, im)
        cv2.waitKey(1)
        # send image through model
        if self.model is not None:
            image_tensor = None
            if self.model_type == "CNN":
                image_tensor = self.transformCNN(image)
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            elif self.model_type == "RNN":
                #Apply transformation for RNN
                ...
            if image_tensor is not None:
                prediction = self.model(image_tensor)
                self.pub.publish(prediction.item())

    def apply_preprocessing_cnn(self, image):
        """
        Apply preprocessing transformations to the input image only for the CNN network

        Parameters:
        - image: PIL Image object.
        returns np array of preprocessed image
        """

        image_array = np.array(image)
        channels = [image_array[:, :, i] for i in range(image_array.shape[2])]
        h, w, _ = image_array.shape

        imghsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

        img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        mask_ground = np.ones(img.shape, dtype=np.uint8)  # Start with a mask of ones (white)
        mid_point = img.shape[0] // 2  # Integer division to get the middle row index

        # Set the top half of the image to 0 (black)
        mask_ground[:mid_point-30, :] = 0  # Mask the top half (rows 0 to mid_point-1)

        #gaussian filter
        sigma = 3.5
        img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

        sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
        sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
        # Compute the magnitude of the gradients
        Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
        threshold = 35
        mask_mag = (Gmag > threshold)
            #4 Mask yellow and white

        white_lower_hsv = np.array([0,(0*255)/100,(60*255)/100]) # [0,0,50] - [230,100,255]
        white_upper_hsv = np.array([150,(40*255)/100,(100*255)/100])   # CHANGE ME

        yellow_lower_hsv = np.array([(30*179)/360, (30*255)/100, (30*255)/100])        # CHANGE ME
        yellow_upper_hsv = np.array([(90*179)/360, (110*255)/100, (100*255)/100])  # CHANGE ME
        mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
        mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

        mask_sobelx_pos = (sobelx > 0)
        mask_sobelx_neg = (sobelx < 0)
        mask_sobely_pos = (sobely > 0)
        mask_sobely_neg = (sobely < 0)

        final_mask = mask_ground * mask_mag  * (mask_white + mask_yellow) #* (mask_sobelx_neg * mask_sobely_neg + mask_sobelx_pos* mask_sobely_pos)
        # Convert the NumPy array back to a PIL image
        for channel in channels:
            channel *= final_mask
        filtered_image = np.stack(channels, axis=-1)
    
        return filtered_image

if __name__ == '__main__':
    # create the node
    node = ModelNode(node_name='model_node')
    # keep the process from terminating
    rospy.spin()