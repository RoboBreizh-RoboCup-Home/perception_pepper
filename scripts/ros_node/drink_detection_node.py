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
import perception_utils.transform_utils as tf_utils
from robobreizh_msgs.msg import Person, PersonList
from robobreizh_msgs.srv import shoes_detection
import time

class DrinkDetection():
    def __init__(self, model_name , cameras: nc.NaoqiCameras, VISUAL) -> None:
        
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.conf_threshold = 0.65
        self.nms_threshold = 0.5
        self.model_name = model_name
        self.yolo_drink_detector = YOLOV8(model_name=self.model_name,  _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        self.yolo_person_detector = YOLOV8(model_name="receptionist_320",  _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        
        if self.VISUAL: 
            self.bridge = CvBridge()
            self.pub_cv = rospy.Publisher('/roboBreizh_detector/drink_detection_image', Image, queue_size=10)
        
        self.init_service()
        rospy.spin()

    def init_service(self):
        rospy.Service('/robobreizh/perception_pepper/drink_detection',
                        shoes_detection, self.handle_service)
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Drink Detection. "+bcolors.ENDC)
        
        rospy.spin()
        
    def handle_service(self, drink_detection):
        
        ori_rgb_image_320, ori_depth_image = self._cameras.get_image(out_format="cv2")
        
        # Detect person
        # Create Person list
        person_list = PersonList()
        person_list.person_list = []
        person_detections = self.yolo_person_detector.inference(ori_rgb_image_320)
        if (len(person_detections) > 0):
            for i in range(len(person_detections)):
                if (person_detections[i]['class_name'] == 'person'):
                    person_start_x = round(person_detections[i]['box'][0])
                    person_end_x = round((person_detections[i]['box'][0] + person_detections[i]['box'][2]))
                    person_start_y = round(person_detections[i]['box'][1])
                    person_end_y = round((person_detections[i]['box'][1] + person_detections[i]['box'][3])) 
                                        
                    crop_person = ori_rgb_image_320[person_start_y: person_end_y, person_start_x: person_end_x, :]
                    # Shoe dectection
                    drink_detections = self.yolo_drink_detector.inference(crop_person)
                    # print(drink_detections)
                    if (len(drink_detections) > 0):
                        for j in range(len(drink_detections)):
                            if drink_detections[j]['class_name'] == 'drink':
                                
                                person_dist, person_point_x, person_point_y, person_point_z, _, _ = distances_utils.detectDistanceResolution(
                                        ori_depth_image, person_start_x, person_end_y, person_start_y, person_end_x, resolutionRGB=ori_rgb_image_320.shape[:2][::-1])

                                person_odom_point = tf_utils.compute_absolute_pose([person_point_x, person_point_y, person_point_z])
                                
                                # Create Person object
                                person = Person()
                                # Person attributes
                                person.name.data = f"hold_drink_no_{i+1}"
                                person.distance = person_dist
                                person.coord.x = person_odom_point.x
                                person.coord.y = person_odom_point.y
                                person.coord.z = person_odom_point.z
                                person_list.person_list.append(person)
                                
                                if self.VISUAL:
                                    # Display drink
                                    drink_start_x = drink_detections[i]['box'][0]
                                    drink_start_y = drink_detections[i]['box'][1]
                                    drink_end_x = drink_detections[i]['box'][0] + drink_detections[i]['box'][2]
                                    drink_end_y = drink_detections[i]['box'][1] + drink_detections[i]['box'][3]
                                    cv2.rectangle(crop_person, 
                                                  (int(drink_start_x), int(drink_start_y)) , (int(drink_end_x), int(drink_end_y)), (255,0,0), 2)
                                    ori_rgb_image_320[person_start_y: person_end_y, person_start_x: person_end_x, :] = crop_person
        
        else:
            rospy.loginfo(
                        bcolors.R+"[RoboBreizh - Vision]        No Person Detected. "+bcolors.ENDC) 
            return person_list
        
        
        if len(person_list.person_list) == 0:
            rospy.loginfo(
                    bcolors.R+"[RoboBreizh - Vision]        No Drink Detected. "+bcolors.ENDC)   
                    
        if self.VISUAL:
            self.visualiseRVIZ(ori_rgb_image_320)
    
        return person_list


    def visualiseRVIZ(self, image):
        
        ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pub_cv.publish(ros_image) 
    

if __name__ == "__main__":
    print(sys.version)
    rospy.init_node('drink_detection_node', anonymous=True)
    # VISUAL = True
    # qi_ip ='192.168.50.44'
    
    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')
    
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
    model_name = 'drink_320'
    
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    DrinkDetection(model_name , cameras, VISUAL)