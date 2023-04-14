#!/usr/bin/env python3

import rospy
import ros_numpy
import std_msgs.msg
import sensor_msgs.msg
import geometry_msgs.msg

import cv2
import jetson.inference
import jetson.utils
import numpy as np
from math import *


class detector():
    def __init__(self):
        self.image_sub = rospy.Subscriber("/color_image", sensor_msgs.msg.Image, self.callback)
        self.alpha_pub = rospy.Publisher('/alpha', std_msgs.msg.Float32, queue_size=10)
        self.center_bb_pub = rospy.Publisher('/center_boundingbox', geometry_msgs.msg.Point, queue_size=10)
        self.is_person_pub = rospy.Publisher('/is_person', std_msgs.msg.Int16, queue_size=10)
        self.frame = None

    def callback(self, image_data):
        self.frame = np.frombuffer(image_data.data, dtype=np.uint8).reshape(image_data.height, image_data.width, -1)
        self.detect()
        self.frame = None

    def detect(self):
        frame_cuda = jetson.utils.cudaFromNumpy(self.frame)
        detections = net.Detect(frame_cuda)
        center_bb = geometry_msgs.msg.Point()
        is_person = 0

        for detect in detections:
            classID = detect.ClassID
            className = net.GetClassDesc(classID)
            conf = detect.Confidence
            if className == 'person':
                is_person = 1
                x = int(detect.Left)
                y = int(detect.Top)
                w = int(detect.Width)
                h = int(detect.Height)
                area = w * h
                # cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv2.putText(self.frame, str(round(conf, 1)), (x, y - 10), font, 1, (0, 255, 0), 1)
                
                Cx = int(self.frame.shape[1] // 2)
                Cy = int(self.frame.shape[0] // 2)
                Cx_bb = int(x + w//2)
                Cy_bb = int(y + h//2)
                f = 210.831
                center_bb.x = Cx_bb
                center_bb.y = Cy_bb - 80
                center_bb.z = 0
                alpha = -1*(degrees(asin((self.frame.shape[1]-Cx_bb-Cx) / sqrt(f**2 + (self.frame.shape[1]-Cx_bb-Cx)**2 + (Cy_bb-Cy)**2))))

                # Publish data
                self.alpha_pub.publish(alpha)
                self.center_bb_pub.publish(center_bb)
                self.is_person_pub.publish(is_person)
                # print(alpha)
                # print(center_bb)

        self.is_person_pub.publish(is_person)


        # cv2.putText(self.frame, str(round(net.GetNetworkFPS(), 1)) + " fps", (0, 30), font, 1, (0, 0, 255), 1)
        # cv2.putText(self.frame, "Detected " + str(len(detections)) + " human in image", (300, 30), font, 1, (0, 0, 255), 1)
        # if len(confs) != 0:
        #     acc += sum(confs) / len(confs)
                
        # cv2.imshow("result", self.frame)
        # cv2.waitKey(10)

if __name__ == "__main__":
    rospy.init_node('detect', anonymous=True)
    # load the object detection network
    net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
    font = cv2.FONT_HERSHEY_SIMPLEX

    try:
        detector()
        rospy.spin()
    except Exception as e:
        print(e)
        pass
