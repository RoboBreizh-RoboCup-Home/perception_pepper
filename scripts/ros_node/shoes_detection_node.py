#!/usr/bin/env python

# import types
from typing import List

# import roslib
import rospy

# import robobreizh msgs
import Camera.Naoqi_camera as nc
import Camera.naoqi_camera_types as camera_types
from Camera.naoqi_camera_types import CameraID, CameraResolution2D as res2D, CameraResolution3D as res3D, ColorSpace2D as cs2D, ColorSpace3D as cs3D
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import cv2
import numpy as np
import sys

# import utils
from models.ObjectDetection.YOLOV8.yolov8 import YOLOV8
from perception_utils.bcolors import bcolors
import perception_utils.distances_utils as distances_utils
from robobreizh_msgs.msg import *
from robobreizh_msgs.srv import *
import time

class ShoesDetection():
    def __init__(self, model_name , cameras: nc.NaoqiCameras, VISUAL) -> None:
        
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.conf_threshold = 0.65
        self.nms_threshold = 0.5
        self.model_name = model_name
        self.yolo_shoes_detector = YOLOV8(model_name=self.model_name,  _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        self.yolo_person_detector = YOLOV8(model_name="receptionist_320",  _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        
        if self.VISUAL: 
            self.bridge = CvBridge()
            self.pub_cv = rospy.Publisher('/roboBreizh_detector/shoes_detection_image', Image, queue_size=10)
        
        self.initShoesDetectionService()
        rospy.spin()

    def initShoesDetectionService(self):
        rospy.Service('/robobreizh/perception_pepper/shoes_detection',
                        shoes_detection, self.handle_ServiceShoesDetection)
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Shoes Detection. "+bcolors.ENDC)
        
        rospy.spin()
        
    def handle_ServiceShoesDetection(self, shoes_detection):
        
        ori_rgb_image_320, ori_depth_image = self._cameras.get_image(out_format="cv2")
        
        # Detect person
        person_images = []
        detections = self.yolo_person_detector.inference(ori_rgb_image_320)
        if (len(detections)>0):
            for i in range(len(detections)):
                if (detections[i]['class_name'] == 'person') :
                    person_name = detections[i]['class_name']
                    person_start_x = round(detections[i]['box'][0])
                    person_end_x = round((detections[i]['box'][0] + detections[i]['box'][2]))
                    person_start_y = round(detections[i]['box'][1])
                    person_end_y = round((detections[i]['box'][1] + detections[i]['box'][3])) 
                                        
                    crop_person = ori_rgb_image_320[person_start_y: person_end_y, person_start_x: person_end_x, :]
                    person_images.append(crop_person)
        
        
        # Create Chair list object
        shoes_list = ObjectList()
        shoes_list.object_list = []
        # detections = self.yolo_shoes_detector.inference(ori_rgb_image_320)
        
        
        if len(person_images) == 0:
            rospy.loginfo(
                        bcolors.R+"[RoboBreizh - Vision]        No Person Detected. "+bcolors.ENDC) 
            return shoes_list
        
        for person in person_images:
            detections = self.yolo_shoes_detector.inference(person)
        
            print(detections)
            
            if (len(detections) > 0):
                for i in range(len(detections)):
                    if detections[i]['class_name'] == 'footwear':

                        shoes_start_x = detections[i]['box'][0]
                        shoes_start_y = detections[i]['box'][1]
                        shoes_end_x = detections[i]['box'][0] + detections[i]['box'][2]
                        shoes_end_y = detections[i]['box'][1] + detections[i]['box'][3]

                        dist, point_x, point_y, point_z, _, _ = distances_utils.detectDistanceResolution(
                                    ori_depth_image, shoes_start_x, shoes_end_y, shoes_start_y, shoes_end_x , [ori_rgb_image_320.shape[1], ori_rgb_image_320.shape[0]])
                        
                        shoes = Object()
                        shoes.label = String("Shoes")
                        shoes.distance = dist
                        shoes.coord.x = point_x
                        shoes.coord.y = point_y
                        shoes.coord.z = point_z
                        shoes_list.object_list.append(shoes)
                        
                        if self.VISUAL:
                            # Display shoes
                            cv2.rectangle(person, (int(shoes_start_x), int(shoes_start_y)) , (int(shoes_end_x), int(shoes_end_y)), (255,0,0), 2)
                
            else:
                rospy.loginfo(
                        bcolors.R+"[RoboBreizh - Vision]        No Shoes Detected. "+bcolors.ENDC)   
                
            if self.VISUAL:
                self.visualiseRVIZ(person)
    
        return shoes_list


    def visualiseRVIZ(self, image):
        
        ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pub_cv.publish(ros_image) 
    

if __name__ == "__main__":
    print(sys.version)
    rospy.init_node('shoes_detection_node', anonymous=True)
    # VISUAL = True
    # qi_ip ='192.168.50.44'
    
    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')
    
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
    model_name = 'shoes_320'
    
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    ShoesDetection(model_name , cameras, VISUAL)