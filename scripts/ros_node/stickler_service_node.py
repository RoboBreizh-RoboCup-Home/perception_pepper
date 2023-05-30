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
from robobreizh_msgs.msg import Person, PersonList, Object
from robobreizh_msgs.srv import person_features_detection_service
import time

class RuleStickler():
    def __init__(self, 
                 person_model,
                 shoes_model,
                 drink_model, 
                 cameras: nc.NaoqiCameras, 
                 VISUAL) -> None:
        
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.distanceMax = 0
        self.conf_threshold = 0.65
        self.nms_threshold = 0.5
        self.yolo_person_detector = YOLOV8(model_name=person_model, _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        self.yolo_shoes_detector = YOLOV8(model_name=shoes_model, _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        self.yolo_drink_detector = YOLOV8(model_name=drink_model, _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        
        if self.VISUAL: 
            self.bridge = CvBridge()
            self.pub_cv = rospy.Publisher('/roboBreizh_detector/stickler_detection_image', Image, queue_size=10)
        
        self.initShoesDetectionService()
        rospy.spin()

    def init_service(self):
        rospy.Service('/robobreizh/perception_pepper/stickler_service',
                        person_features_detection_service, self.handle_service)
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Rule Stickler Service."+bcolors.ENDC)
        
        rospy.spin()
        
    def handle_service(self, person_request):
        
        self.distanceMax = person_request.entries_list.distanceMaximum
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
                    
                    
                    person_dist, person_point_x, person_point_y, person_point_z, _, _ = distances_utils.detectDistanceResolution(
                                        ori_depth_image, person_start_x, person_end_y, person_start_y, person_end_x, resolutionRGB=ori_rgb_image_320.shape[:2][::-1])

                    if person_dist > self.distanceMax:
                        rospy.loginfo(
                            bcolors.R+"[RoboBreizh - Vision]        Person Detected but not within range. "+bcolors.ENDC)
                        continue
                    
                    person_odom_point = tf_utils.compute_absolute_pose([person_point_x, person_point_y, person_point_z])
                    
                    # Create Person object
                    person = Person()
                    # Person attributes
                    person.name.data = f"person_no_{i+1}"
                    person.distance = person_dist
                    person.coord.x = person_odom_point.x
                    person.coord.y = person_odom_point.y
                    person.coord.z = person_odom_point.z
                    person_list.person_list.append(person)
                    
                    crop_person = ori_rgb_image_320[person_start_y: person_end_y, person_start_x: person_end_x, :]
                    # Shoe dectection
                    shoes_detection_result = self.perform_detection(self.yolo_shoes_detector, crop_person, crop_person.copy())
                    person.is_shoes = shoes_detection_result["is_detected"]
                    draw_image = shoes_detection_result["draw_image"]
                    if not person.is_shoes:
                        rospy.loginfo(
                            bcolors.R+"[RoboBreizh - Vision]        No Shoes Detected. "+bcolors.ENDC)   
                    
                    # Drink detection
                    drink_detection_result = self.perform_detection(self.yolo_drink_detector, crop_person, draw_image,
                                                                    [person_start_x, person_end_y, person_start_y, person_end_x],
                                                                    ori_rgb_image_320, ori_depth_image)
                    person.is_drink = drink_detection_result["is_detected"]
                    person.drink = drink_detection_result["drink_info"]
                    draw_image = drink_detection_result["draw_image"]
                    if not person.is_drink:
                        rospy.loginfo(
                            bcolors.R+"[RoboBreizh - Vision]        No Drink Detected. "+bcolors.ENDC)

                    if self.VISUAL:
                        ori_rgb_image_320[person_start_y: person_end_y, person_start_x: person_end_x, :] = draw_image
        
        else:
            rospy.loginfo(
                        bcolors.R+"[RoboBreizh - Vision]        No Person Detected. "+bcolors.ENDC) 
            return person_list
            
                    
        if self.VISUAL:
            ros_image = self.bridge.cv2_to_imgmsg(ori_rgb_image_320, "bgr8")
            self.pub_cv.publish(ros_image) 
    
        return person_list
        
    
    def perform_detection(self, model, person_image, draw_image,
                          person_coord=None, ori_rgb_image=None, ori_depth_image=None):
        
        obj = Object()
        return_dict = {
            "is_detected": False,
            "draw_image": draw_image,
            "drink_info": obj
        }
        detected_classes = ["footwear", "drink"]
        detections = model.inference(person_image)
        if (len(detections) > 0):
            for i in range(len(detections)):
                object_name = detections[i]['class_name']
                if object_name in detected_classes:
                    return_dict["is_detected"] = True
                    
                    if person_coord:
                        dist, point_x, point_y, point_z, _, _ = distances_utils.detectDistanceResolution(
                                ori_depth_image, 
                                start_x + person_coord[0], end_y + person_coord[1], 
                                start_y + person_coord[2], end_x + person_coord[3], 
                                resolutionRGB=ori_rgb_image.shape[:2])
                        
                        odom_point = tf_utils.compute_absolute_pose([point_x, point_y, point_z])  
                        # drink attribute
                        return_dict["drink_info"].label = String(object_name)
                        return_dict["drink_info"].distance = dist
                        return_dict["drink_info"].coord.x = odom_point.x
                        return_dict["drink_info"].coord.y = odom_point.y
                        return_dict["drink_info"].coord.z = odom_point.z
                        
                    if self.VISUAL:
                        # Display shoes
                        start_x = detections[i]['box'][0]
                        start_y = detections[i]['box'][1]
                        end_x = detections[i]['box'][0] + detections[i]['box'][2]
                        end_y = detections[i]['box'][1] + detections[i]['box'][3]
                        cv2.rectangle(draw_image, 
                                      (int(start_x), int(start_y)) , (int(end_x), int(end_y)), (255,0,0), 2)
        
        return_dict["draw_image"] = draw_image   
        return return_dict
    

if __name__ == "__main__":
    print(sys.version)
    rospy.init_node('shoes_detection_node', anonymous=True)
    # VISUAL = True
    # qi_ip ='192.168.50.44'
    
    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')
    
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
    person_model_name = 'receptionist_320'
    shoes_model_name = 'shoes_320'
    drink_model_name = 'drink_320'
    
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    RuleStickler(person_model_name, shoes_model_name, drink_model_name , cameras, VISUAL)