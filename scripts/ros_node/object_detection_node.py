#!/usr/bin/env python

# import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import Camera.Naoqi_camera as nc
from Camera.naoqi_camera_types import CameraID, CameraResolution2D as res2D, CameraResolution3D as res3D, ColorSpace2D as cs2D, ColorSpace3D as cs3D
from cv_bridge import CvBridge
import cv2
import numpy as np

# import utils
from perception_utils.bcolors import bcolors
import perception_utils.distances_utils as distances_utils
import perception_utils.transform_utils as tf_utils
from perception_utils.objects_detection_utils import *
from perception_utils.utils import get_pkg_path
from models.ObjectDetection.YOLOV8.yolov8 import YOLOV8
from robobreizh_msgs.msg import *
from robobreizh_msgs.srv import *
import tf2_ros
import time

class ObjectDetection():
        
    def __init__(self , model_name, cameras: nc.NaoqiCameras, VISUAL) -> None:
        
        #self.VISUAL :bool = rospy.get_param('~visualize')
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.model_name = model_name
        self.conf_threshold = 0.3
        self.nms_threshold = 0.5
        self.object_requested_list = []
        self.distanceMax = 0
        self.pkg_path = get_pkg_path()
        self.yolo_detector = YOLOV8(model_name=self.model_name, _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]    Loading Object Detection weights done"+bcolors.ENDC)
        
        if self.VISUAL: 
            self.bridge = CvBridge()
            # self.pub_opencv = rospy.Publisher('/roboBreizh_detector/object_detection_raw_image', Image, queue_size=10)
            self.pub_compressed_img = rospy.Publisher("/roboBreizh_detector/object_detection_compressed_image",
            CompressedImage,  queue_size=10)
        
        self.initObjectDescriptionService()
        
    def initObjectDescriptionService(self):
        rospy.Service('/robobreizh/perception_pepper/object_detection',
                        object_detection_service, self.handle_ServicePerceptionObject)
            
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Objects Detection. "+bcolors.ENDC)
        rospy.spin()

    def handle_ServicePerceptionObject(self, object_detection_service):
        
        objects_Requested = object_detection_service.entries_list
        self.distanceMax = object_detection_service.entries_list.distanceMaximum
        
        for i in range(len(objects_Requested.obj)):
            self.object_requested_list.append(objects_Requested.obj[i].data)
        
        print("Object Requested List: ")
        print(self.object_requested_list)
        print("Distance Maximum: ")
        print(self.distanceMax)
        
        RequestObject = ((self.object_requested_list)[0] != '')
        
        # Create Chair list object
        obj_list= ObjectList()
        obj_list.object_list = []
        
        time_start = time.time()
        
        # retrieve rgb and depth image from Naoqi camera
        ori_rgb_image_320, ori_depth_image = self._cameras.get_image(out_format="cv2")
        detections = self.yolo_detector.inference(ori_rgb_image_320)
        
        if (len(detections)>0):       
            for i in range(len(detections)):
                object_name = detections[i]['class_name']
                if (RequestObject):
                    rospy.loginfo("Detecting Requested objects only")
                    if object_name in self.object_requested_list:
                        start_x = round(detections[i]['box'][0])
                        end_x = round((detections[i]['box'][0] + detections[i]['box'][2]))
                        start_y = round(detections[i]['box'][1])
                        end_y = round((detections[i]['box'][1] + detections[i]['box'][3]))
                    else:
                        rospy.loginfo(
                            bcolors.R+"[RoboBreizh - Vision]        Requested Objects Not in the model class list. "+bcolors.ENDC)   
                        start_x = 0
                        end_x = 0
                        start_y = 0
                        end_y = 0
                else:
                    rospy.loginfo("Detecting all objects")
                    start_x = round(detections[i]['box'][0])
                    end_x = round((detections[i]['box'][0] + detections[i]['box'][2]))
                    start_y = round(detections[i]['box'][1])
                    end_y = round((detections[i]['box'][1] + detections[i]['box'][3]))
                                
                # Distance Detection
                dist, point_x, point_y, point_z, _, _ = distances_utils.detectDistanceResolution(
                        ori_depth_image, start_x, end_y, start_y, end_x, resolutionRGB=[ori_rgb_image_320.shape[1], ori_rgb_image_320.shape[0]])
                
                odom_point = tf_utils.compute_absolute_pose([point_x,point_y,point_z])

                time_end = time.time()
                rospy.loginfo("Total time inference: " + str(time_end-time_start))
                
                if dist < self.distanceMax:
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
                        bcolors.R+"[RoboBreizh - Vision]        Objects Detected but not within range. "+bcolors.ENDC)   
        else:
            rospy.loginfo(
                bcolors.R+"[RoboBreizh - Vision]        No Objects Detected. "+bcolors.ENDC)               
                        
        if self.VISUAL:
            self.visualiseRVIZ(ori_rgb_image_320)
        
        return obj_list
    
    def visualiseRVIZ(self, cv_image):
        
        # ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        # self.pub_opencv.publish(ros_image) 
        
        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tostring()
        # Publish new image
        self.pub_compressed_img.publish(msg)

if __name__ == "__main__":
    
    rospy.init_node('object_detection_node', anonymous=True)

    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')
    
    # VISUAL = True
    # qi_ip = "192.168.50.44"
            
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
 
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    ObjectDetection("ycb_320", cameras, VISUAL)