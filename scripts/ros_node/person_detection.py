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

class PersonDetection():
        
    def __init__(self , model_name, cameras: nc.NaoqiCameras, VISUAL) -> None:
        
        #self.VISUAL :bool = rospy.get_param('~visualize')
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.model_name = model_name
        self.conf_threshold = 0.3
        self.nms_threshold = 0.4
        self.distanceMax = 0
        self.pkg_path = get_pkg_path()
        self.yolo_seat_detector = YOLOV8(model_name=self.model_name, _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]    Loading Person Detection weights done"+bcolors.ENDC)
        
        if self.VISUAL: 
            self.bridge = CvBridge()
            self.pub_opencv = rospy.Publisher('/roboBreizh_detector/person_detection_image', Image, queue_size=10)
            self.pub_compressed_img = rospy.Publisher("/roboBreizh_detector/person_detection_compressed_image",
            CompressedImage,  queue_size=10)
        
        self.initPersonDescriptionService()
        
    def initPersonDescriptionService(self):
        rospy.Service('/robobreizh/perception_pepper/person_detection',
                        person_features_detection_service, self.handle_ServicePerceptionHuman)
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Person Detection. "+bcolors.ENDC)
        rospy.spin()

    def handle_ServicePerceptionHuman(self, person_features_detection_service):
      
        # entries list of maximum distance for person detection
        self.distanceMax = person_features_detection_service.entries_list.distanceMaximum
                
        # Create Person List to store attributes for each person
        person_list = PersonList()
        person_list.person_list = []
    
        time_start = time.time()
        
        # retrieve rgb and depth image from Naoqi camera
        ori_rgb_image, ori_depth_image = self._cameras.get_image(out_format="cv2")

        detections = self.yolo_seat_detector.inference(ori_rgb_image)
        
        if (len(detections)>0):
            for i in range(len(detections)):
                if (detections[i]['class_name'] == 'person') :
                    person_name = detections[i]['class_name']
                    person_start_x = round(detections[i]['box'][0])
                    person_end_x = round((detections[i]['box'][0] + detections[i]['box'][2]))
                    person_start_y = round(detections[i]['box'][1])
                    person_end_y = round((detections[i]['box'][1] + detections[i]['box'][3])) 
                                        
                    person_bounding_box = [person_start_x, person_start_y, person_end_x, person_end_y]

                    # Distance Detection
                    person_dist, person_point_x, person_point_y, person_point_z, _, _ = distances_utils.detectDistanceResolution(
                            ori_depth_image, person_start_x, person_end_y, person_start_y, person_end_x, resolutionRGB=[ori_rgb_image.shape[1], ori_rgb_image.shape[0]])

                    person_odom_point = tf_utils.compute_absolute_pose([person_point_x,person_point_y,person_point_z])
                    
                    if person_dist <= self.distanceMax:
                    
                        # Create Person object
                        person = Person()
                        # Person attributes
                        person.distance = person_dist
                        person.coord.x = person_odom_point.x
                        person.coord.y = person_odom_point.y
                        person.coord.z = person_odom_point.z
                        person_list.person_list.append(person)
                    
                    if self.VISUAL:
                        cv2.rectangle(ori_rgb_image, (person_start_x, person_start_y), (person_end_x,person_end_y), (255,0,0), 0)
        else:
            rospy.loginfo(
                bcolors.R+"[RoboBreizh - Vision]        No Objects Detected. "+bcolors.ENDC)               
                        
        if self.VISUAL: 
            self.visualiseRVIZ(ori_rgb_image)

        time_end = time.time()
        rospy.loginfo("Total time inference: " + str(time_end-time_start))
        
        return person_list 
    
    def visualiseRVIZ(self, chair_image):
        
        cv_chair_image = self.bridge.cv2_to_imgmsg(chair_image, "bgr8")
        self.pub_opencv.publish(cv_chair_image) 

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', chair_image)[1]).tostring()
        # Publish new image
        self.pub_compressed_img.publish(msg)

if __name__ == "__main__":
    
    rospy.init_node('person_detection_node', anonymous=True)

    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')
    
    # VISUAL = True
    # qi_ip = "192.168.50.44"
    
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
 
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    PersonDetection("receptionist_320", cameras, VISUAL)