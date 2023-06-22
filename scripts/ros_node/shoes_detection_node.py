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
from robobreizh_msgs.msg import Object, ObjectList
from robobreizh_msgs.srv import shoes_detection
import time

class ShoesDetection():
    def __init__(self, model_name , cameras: nc.NaoqiCameras, VISUAL) -> None:
        
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.conf_threshold = 0.65
        self.nms_threshold = 0.5
        self.distanceMax = 0
        self.model_name = model_name
        self.yolo_shoes_detector = YOLOV8(model_name=self.model_name,  _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        
        if self.VISUAL: 
            self.bridge = CvBridge()
            self.pub_cv = rospy.Publisher('/roboBreizh_detector/shoes_detection_image', Image, queue_size=10)
        
        self.init_service()
        rospy.spin()

    def init_service(self):
        rospy.Service('/robobreizh/perception_pepper/shoes_detection',
                        shoes_detection, self.handle_service)
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Shoes Detection. "+bcolors.ENDC)
        
        rospy.spin()
        
    def handle_service(self, shoes_detection):
        
        ori_rgb_image_320, ori_depth_image = self._cameras.get_image(out_format="cv2")
        
        self.distanceMax = shoes_detection.entries_list.distanceMaximum
        obj_list= ObjectList()
        obj_list.object_list = []
        
        time_start = time.time()
        
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

                    obj = Object()
                        
                    # Chair attributes
                    obj.label = String(object_name)
                    obj.distance = dist
                    obj.coord.x = odom_point.x
                    obj.coord.y = odom_point.y
                    obj.coord.z = odom_point.z
                    
                    obj_list.object_list.append(obj)
                    
        else:
            rospy.loginfo(
                    bcolors.R+"[RoboBreizh - Vision]        No Shoes Detected. "+bcolors.ENDC)  
        
                    
        if self.VISUAL:
            self.visualiseRVIZ(ori_rgb_image_320)
    
        return obj_list


    def visualiseRVIZ(self, image):
        
        ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pub_cv.publish(ros_image) 
    

if __name__ == "__main__":
    
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