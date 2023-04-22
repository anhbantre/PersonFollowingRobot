#!/usr/bin/env python3

import rospy
import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg

import cv2
import numpy as np
import jetson.utils
import jetson.inference
from math import *

FOCAL_DIST = 210.831

class Detector():
    """
    This class is used to detect people in an image
    """
    def __init__(self):
        """
        Initializes the subscribers and publishers for the ROS topics
        """
        self.image_sub = rospy.Subscriber("/color_image", sensor_msgs.msg.Image, self.callback)
        self.alpha_pub = rospy.Publisher('/alpha', std_msgs.msg.Float32, queue_size=10)
        self.center_bb_pub = rospy.Publisher('/center_boundingbox', geometry_msgs.msg.Point, queue_size=10)
        self.is_person_pub = rospy.Publisher('/is_person', std_msgs.msg.Int16, queue_size=10)
        self.frame = None

    def callback(self, image_data):
        """
        Callback function that is called each time a new image is received

        Args:
            image_data (sensor_msgs.msg.Image): The received image message
        """
        # Convert the image message to an OpenCV image
        self.frame = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
        # Run the detector
        self.detect()
        # Free memory
        self.frame = None

    def detect(self):
        # Covert the image from numpy array to jetson cuda
        frame_cuda = jetson.utils.cudaFromNumpy(self.frame)
        # Inference
        detections = net.Detect(frame_cuda)
        # Create a Point message
        center_bb = geometry_msgs.msg.Point()
        is_person = 0

        for detection  in detections:
            class_id = detection.ClassID
            class_name = net.GetClassDesc(class_id)
            confidence = detection.Confidence
            if class_name == 'person':
                is_person = 1
                # Get bounding box coordinates and convert them to int values
                x, y, w, h = map(int, (detection.Left, detection.Top, detection.Width, detection.Height))
                
                # Calculate the center of the bounding box
                Cx = int(self.frame.shape[1] // 2)
                Cy = int(self.frame.shape[0] // 2)
                Cx_bb = int(x + w//2)
                Cy_bb = int(y + h//2)
                center_bb.x = Cx_bb
                center_bb.y = Cy_bb - 80    # 80 is the distance from the center of the bounding box to the person to make sure the person is in front of the robot
                center_bb.z = 0

                # Calculate the alpha value which is the angle between the robot and the object
                alpha = -1*(degrees(asin((self.frame.shape[1]-Cx_bb-Cx) / sqrt(FOCAL_DIST**2 + (self.frame.shape[1]-Cx_bb-Cx)**2 + (Cy_bb-Cy)**2))))

                # Publish data
                self.alpha_pub.publish(alpha)
                self.center_bb_pub.publish(center_bb)
                self.is_person_pub.publish(is_person)

        self.is_person_pub.publish(is_person)

if __name__ == "__main__":
    rospy.init_node('detect', anonymous=True)
    # load the object detection network
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        Detector()
        rospy.spin()
    except Exception as e:
        print(e)
        pass
