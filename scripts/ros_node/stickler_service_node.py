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
from robobreizh_msgs.msg import Person, PersonList, Object, ObjectList
from robobreizh_msgs.srv import person_features_detection_service, shoes_detection
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
        self.drink_threshold = 0.3
        self.nms_threshold = 0.5
        self.yolo_person_detector = YOLOV8(model_name=person_model, _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        self.yolo_shoes_detector = YOLOV8(model_name=shoes_model, _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        self.yolo_drink_detector = YOLOV8(model_name=drink_model, _conf_threshold=self.drink_threshold, _iou_threshold=self.nms_threshold)
        
        if self.VISUAL: 
            self.bridge = CvBridge()
            self.pub_cv = rospy.Publisher('/roboBreizh_detector/stickler_detection_image', Image, queue_size=10)
            self.pub_cv_shoes = rospy.Publisher('/roboBreizh_detector/shoes_detection_image', Image, queue_size=10)

        self.init_service()
        
        rospy.spin()

    def init_service(self):
        rospy.Service('/robobreizh/perception_pepper/stickler_service',
                        person_features_detection_service, self.handle_service)
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Rule Stickler Service."+bcolors.ENDC)

        rospy.Service('/robobreizh/perception_pepper/shoes_detection',
                        shoes_detection, self.handle_service_shoes)
        
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Shoes Detection. "+bcolors.ENDC)
     
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

                    if person_dist <= self.distanceMax:
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
                        shoes_detection_result = self.shoes_drink_inference(self.yolo_shoes_detector, crop_person, 
                                                                        [person_start_x, person_start_y, person_end_x, person_end_y],
                                                                        ori_rgb_image_320, ori_depth_image)
                        person.is_shoes = shoes_detection_result["is_detected"]
                        
                        if not person.is_shoes:
                            rospy.loginfo(
                                bcolors.R+"[RoboBreizh - Vision]        Human is not wearing any Shoes. "+bcolors.ENDC)   
                        
                        # Drink detection
                        drink_detection_result = self.shoes_drink_inference(self.yolo_drink_detector, crop_person,
                                                                        [person_start_x, person_start_y, person_end_x, person_end_y],
                                                                        ori_rgb_image_320, ori_depth_image)
                        
                        person.is_drink = drink_detection_result["is_detected"]
                        
                        if not person.is_drink:
                            rospy.loginfo(
                                bcolors.R+"[RoboBreizh - Vision]        Human is not holding any Drink. "+bcolors.ENDC)

                        if self.VISUAL:
                            cv2.rectangle(ori_rgb_image_320, 
                                      (int(person_start_x), int(person_start_y)) , (int(person_end_x), int(person_end_y)), (0,255,255), 2)
                            
                    else:
                        rospy.loginfo(
                            bcolors.R+"[RoboBreizh - Vision]        Person Detected but not within range. "+bcolors.ENDC)
        
        else:
            rospy.loginfo(
                bcolors.R+"[RoboBreizh - Vision]        No Person Detected. "+bcolors.ENDC) 
            
            return person_list
            
                    
        if self.VISUAL:
            ros_image = self.bridge.cv2_to_imgmsg(ori_rgb_image_320, "bgr8")
            self.pub_cv.publish(ros_image) 
    
        return person_list
        

        
    def handle_service_shoes(self, shoes_detection):
        
        ori_rgb_image_320, ori_depth_image = self._cameras.get_image(out_format="cv2")
        
        self.distanceMax = shoes_detection.distance_max
        person_list = PersonList()
        person_list.person_list = []
        
        time_start = time.time()
        
        shoes_on = True
        
        ori_rgb_image_320, ori_depth_image = self._cameras.get_image(out_format="cv2")
        
        detections = self.yolo_shoes_detector.inference(ori_rgb_image_320)
        if (len(detections) > 0):
            for i in range(len(detections)):
                object_name = detections[i]['class_name']
                if object_name == 'footwear':
                    start_x = round(detections[i]['box'][0])
                    end_x = round((detections[i]['box'][0] + detections[i]['box'][2]))
                    start_y = round(detections[i]['box'][1])
                    end_y = round((detections[i]['box'][1] + detections[i]['box'][3]))
                                
                    # Distance Detection
                    dist, point_x, point_y, point_z, _, _ = distances_utils.detectDistanceResolution(
                            ori_depth_image, start_x, end_y, start_y, end_x, resolutionRGB=ori_rgb_image_320.shape[:2][::-1])
                    
                    odom_point = tf_utils.compute_absolute_pose([point_x,point_y,point_z])

                    time_end = time.time()
                    rospy.loginfo("Total time inference: " + str(time_end-time_start))
                    
                    if dist > self.distanceMax:
                        rospy.loginfo(
                            bcolors.R+"[RoboBreizh - Vision]        Shoes Detected but not within range. "+bcolors.ENDC)
                        continue
                    
                    if self.VISUAL:
                        cv2.rectangle(ori_rgb_image_320, (start_x, start_y), (end_x,end_y), (255,255,0), 0)

                    person = Object()
                        
                    # Chair attributes
                    person.label = String(object_name)
                    person.distance = dist
                    person.is_shoes = shoes_on
                    
                    person_list.object_list.append(person)
                    
        else:
            rospy.loginfo(
                    bcolors.R+"[RoboBreizh - Vision]        No Shoes Detected. "+bcolors.ENDC)  
        
        if self.VISUAL:
            ros_image_shoes = self.bridge.cv2_to_imgmsg(ori_rgb_image_320, "bgr8")
            self.pub_cv_shoes.publish(ros_image_shoes) 
    
        return person_list

    
    def shoes_drink_inference(self, model, person_image,
                          person_coord=None, ori_rgb_image=None, ori_depth_image=None):
        return_dict = {
            "is_detected": False,
        }
        detected_classes = ["footwear", "drink"]
        detections = model.inference(person_image)
        if (len(detections) > 0):
            for i in range(len(detections)):
                object_name = detections[i]['class_name']
                if object_name in detected_classes:
                    return_dict["is_detected"] = True
                    start_x = detections[i]['box'][0]
                    start_y = detections[i]['box'][1]
                    end_x = detections[i]['box'][0] + detections[i]['box'][2]
                    end_y = detections[i]['box'][1] + detections[i]['box'][3]
                    
                    if person_coord:
                        
                        start_x_person_obj = (start_x + person_coord[0])
                        start_y_person_obj = (start_y + person_coord[1])
                        end_x_person_obj = (end_x + person_coord[0])
                        end_y_person_obj = (end_y + person_coord[1])
                        
                    if self.VISUAL:
                        # Display detection results
                        cv2.rectangle(ori_rgb_image, 
                                      (int(start_x_person_obj), int(start_y_person_obj)) , (int(end_x_person_obj), int(end_y_person_obj)), (255,255,0), 2)
        
        return return_dict
    

    def visualiseRVIZ(self, image):
        
        ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pub_cv_shoes.publish(ros_image) 

if __name__ == "__main__":
    print(sys.version)
    rospy.init_node('rule_stickler_node', anonymous=True)
    # VISUAL = True
    # qi_ip ='192.168.50.44'
    
    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')
    
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
    person_model_name = 'receptionist_320'
    shoes_model_name = 'shoes_320'
    drink_model_name = 'drinks_320'
    
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    RuleStickler(person_model_name, shoes_model_name, drink_model_name , cameras, VISUAL)