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
import time
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
        self.i = 0
      
        self.model_type = "CNN2"
        self.model = None
        
        if self.model_type == "CNN":
            self.model = LaneDetectionCNN((3, 224, 224))
            self.load_model("packages/pidplus/src/model/latest_cnn.pth")

        elif self.model_type == "CNN2":
            self.model = LaneDetectionCNN((3, 480, 640))
            self.load_model("packages/pidplus/src/model/latest_cnn2.pth")

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

        self.transformCNN2 = transforms.Compose([
        transforms.Lambda(self.apply_preprocessing_cnn2),
        transforms.Lambda(Image.fromarray),
        transforms.ToTensor(),  # Convert image to tensor
        ])
        self.last_time = 0
        self.last_image = None
          # construct subscriber
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        # create publisher
        self.pub = rospy.Publisher('prediction', Float32, queue_size=10)
        
    def load_model(self, model_path):
        # Load the model
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.model.eval()
    
    def run(self):
        while not rospy.is_shutdown():
            image = self.last_image
            # send image through model
            if self.model is not None and image is not None:
                image_tensor = None
                if self.model_type == "CNN":
                    image_tensor = self.transformCNN(image)
                    #im = image_tensor.permute(1, 2, 0).cpu().numpy()
                    #cv2.imshow(self._window, im)
                    #cv2.waitKey(1)
                    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                elif self.model_type =="CNN2":
                    image_tensor = self.transformCNN2(image)
                    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
                elif self.model_type == "RNN":
                    #Apply transformation for RNN
                    ...
                if image_tensor is not None:
                    prediction = self.model(image_tensor)
                    self.pub.publish(prediction.item())

    def callback(self, msg):
        # convert JPEG bytes to CV image
        image = self._bridge.compressed_imgmsg_to_cv2(msg)
        im = image

        self.last_image = image
        #self.last_time = time.time()
        #if self.model_type == "CNN":
        #    im = self.apply_preprocessing_cnn(image)
        #elif self.model_type == "CNN2":
        #    im = self.apply_preprocessing_cnn2(image)
#
        #cv2.imshow(self._window, im)
        #cv2.waitKey(1)

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
    def apply_preprocessing_cnn2(self, image):
        """
        Apply preprocessing transformations to the input image.

        Parameters:
        - image: PIL Image object.
        """

        image_array = np.array(image)

        blurred_image_array = cv2.GaussianBlur(image_array, (0, 0), 0.1)
        channels = [image_array[:, :, i] for i in range(image_array.shape[2])]
        h, w, _ = image_array.shape

        imghsv = cv2.cvtColor(blurred_image_array, cv2.COLOR_BGR2HSV)
        img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

        mask_ground = np.ones(img.shape, dtype=np.uint8)  # Start with a mask of ones (white)


        one_third_height = h // 3
        # crop_height = h * 2 // 5 
        mask_ground[:one_third_height, :] = 0  # Mask the top 1/3 of the image

        #gaussian filter
        sigma = 4.5
        img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)

        sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
        sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
        # Compute the magnitude of the gradients
        Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
        threshold = 51


        white_lower_hsv = np.array([0, 0, 143])         # CHANGE ME
        white_upper_hsv = np.array([228, 60, 255])   # CHANGE ME
        yellow_lower_hsv = np.array([10, 50, 100])        # CHANGE ME
        yellow_upper_hsv = np.array([70, 255, 255])  # CHANGE ME

        mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
        mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)

        # crop two fifth of the image on the right for the yello mask
        height, width = mask_yellow.shape 
        crop_width = width * 2 // 5 
        crop_width_2 = width * 1 // 2 
        crop_mask_11 = np.zeros_like(mask_yellow, dtype=np.uint8)
        crop_mask_11[:, :width - crop_width_2] = 1 
        mask_yellow = mask_yellow * crop_mask_11

        # crop two fifth of the image on the left for the white mask
        crop_mask_22 = np.zeros_like(mask_white, dtype=np.uint8)
        crop_mask_22[:, crop_width:] = 1 
        mask_white = mask_white * crop_mask_22


        mask_mag = (Gmag > threshold)

        # np.savetxt("mask.txt", mask_white, fmt='%d', delimiter=',')
        # exit()
        crop_width_3 = width * 1 // 10 
        crop_mask_33 = np.zeros_like(mask_yellow, dtype=np.uint8)
        crop_mask_33[:, :width - crop_width_3] = 1 

        crop_mask_44 = np.zeros_like(mask_white, dtype=np.uint8)
        crop_mask_44[:, crop_width_3:] = 1 

        final_mask = mask_ground * mask_mag * 255 
        mask_white = mask_ground * mask_white
        mask_yellow = mask_ground * mask_yellow
        # Convert the NumPy array back to a PIL image

        channels[0] =  final_mask #np.zeros_like(channels[0])
        channels[1] =  mask_white
        channels[2] =  mask_yellow

        filtered_image = np.stack(channels, axis=-1)
        #filtered_image = Image.fromarray(filtered_image)
        return  filtered_image

if __name__ == '__main__':
    # create the node
    node = ModelNode(node_name='model_node')

    node.run()
    # keep the process from terminating
    rospy.spin()