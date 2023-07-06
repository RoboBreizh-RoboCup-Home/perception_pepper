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

class CategoryDetection():
        
    def __init__(self , model_name, cameras: nc.NaoqiCameras, VISUAL) -> None:
        
        #self.VISUAL :bool = rospy.get_param('~visualize')

        self.shelf_height = {'cabinet1' : [1.06, 0.73],
                            'cabinet2': [0.73, 0.45], 
                            'cabinet3': [0.45, 0.1]}
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.model_name = model_name
        self.conf_threshold = 0.4
        self.nms_threshold = 0.5
        self.object_requested_list = []
        self.distanceMax = 0
        self.pkg_path = get_pkg_path()
        self.yolo_detector = YOLOV8(model_name=self.model_name, _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]    Loading Object Detection (Category) weights done"+bcolors.ENDC)
        
        if self.VISUAL: 
            self.bridge = CvBridge()
            self.pub_opencv = rospy.Publisher('/roboBreizh_detector/object_category_raw_image', Image, queue_size=10)
            rng = np.random.default_rng(3)
            self.colors = rng.uniform(0, 255, size=(len(self.yolo_detector.classes), 3))

            self.pub_compressed_img = rospy.Publisher("/roboBreizh_detector/object_category_compressed_image",
            CompressedImage,  queue_size=10)
        
        self.initObjectDescriptionService()
        
    def initCategoryDescriptionService(self):
        rospy.Service('/robobreizh/perception_pepper/category_detection',
                        category_detection_service, self.handle_ServicePerceptionCategory)
            
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Objects Detection (Category). "+bcolors.ENDC)
        rospy.spin()

    def handle_ServicePerceptionCategory(self, object_detection_service):
        
        objects_Requested = object_detection_service.entries_list
        self.distanceMax = object_detection_service.entries_list.distanceMaximum
        self.shelf = object_detection_service.shelf_name
        
        for i in range(len(objects_Requested.obj)):
            self.object_requested_list.append((objects_Requested.obj[i].data).lower())
        
        print("Object Requested List: ")
        print(self.object_requested_list)
        print("Distance Maximum: ")
        print(self.distanceMax)
        
        RequestObject = ((self.object_requested_list)[0] != '')
        
        # Create Object list
        obj_list= ObjectList()
        obj_list.object_list = []
        
        time_start = time.time()
        
        # retrieve rgb and depth image from Naoqi camera
        ori_rgb_image_320, ori_depth_image = self._cameras.get_image(out_format="cv2")
        detections = self.yolo_detector.inference(ori_rgb_image_320)
        
        time_end = time.time()
        rospy.loginfo("Total time inference: " + str(time_end-time_start))
        
        image_height, image_width = ori_rgb_image_320.shape[0], ori_rgb_image_320.shape[1]
        left_most = ""
        right_most = ""
        if (len(detections)>0):

            left_most = 10000
            right_most = 0
            left_most_object = ""
            right_most_object = ""
            for i in range(len(detections)):
                # Filter everything out of the shelf

                start_x = round(detections[i]['box'][0])
                end_x = round((detections[i]['box'][0] + detections[i]['box'][2]))
                start_y = round(detections[i]['box'][1])
                end_y = round((detections[i]['box'][1] + detections[i]['box'][3]))

                dist, point_x, point_y, point_z, _, _ = distances_utils.detectDistanceResolution(
                        ori_depth_image, start_x, end_y, start_y, end_x, resolutionRGB=[ori_rgb_image_320.shape[1], ori_rgb_image_320.shape[0]])

                if self.shelf_height[self.shelf][1] < point_z < self.shelf_height[self.shelf][1]:
                    if detections[i]['box'][0] < left_most:
                        left_most = detections[i]['box'][0]
                        left_most_object = i
                    if detections[i]['box'][0] > right_most:
                        right_most = detections[i]['box'][0]
                        right_most_object = i
                else:
                    continue

            for i in zip(left_most_object, right_most_object):
                object_name = detections[i]['class_name']

                if self.VISUAL:
                    start_x = round(detections[i]['box'][0])
                    end_x = round((detections[i]['box'][0] + detections[i]['box'][2]))
                    start_y = round(detections[i]['box'][1])
                    end_y = round((detections[i]['box'][1] + detections[i]['box'][3]))

                    size = min([image_height, image_width]) * 0.001
                    text_thickness = int(min([image_height, image_width]) * 0.001)
                    color = self.colors[self.yolo_detector.classes.index(object_name)]
                    cv2.rectangle(ori_rgb_image_320, (start_x, start_y), (end_x,end_y), color, 0)
                    cv2.putText(ori_rgb_image_320, object_name, (start_x, start_y),  cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), text_thickness, cv2.LINE_AA)
            
                    
                    # odom_point = tf_utils.compute_absolute_pose([point_x,point_y,point_z])

                # if dist < self.distanceMax:
                obj = Object()
                    
                obj.label = String(object_name)
                obj.distance = 0.0
                obj.coord.x = 0.0
                obj.coord.y = 0.0
                obj.coord.z = 0.0
                
                obj_list.object_list.append(obj)
                else:
                    continue

        else:
            rospy.loginfo(
                bcolors.R+"[RoboBreizh - Vision]        No Objects Detected. "+bcolors.ENDC)               
                        
        if self.VISUAL:
            self.visualiseRVIZ(ori_rgb_image_320)
        
        return obj_list
    
    def compute_relative_pose(self, detections):
        left_most = 10000
        right_most = 0
        left_most_object = ""
        right_most_object = ""
        for i in range(len(detections)):
            # Filter everything out of the shelf
            dist, point_x, point_y, point_z, _, _ = distances_utils.detectDistanceResolution(
                    ori_depth_image, start_x, end_y, start_y, end_x, resolutionRGB=[ori_rgb_image_320.shape[1], ori_rgb_image_320.shape[0]])

            if self.shelf_height[self.shelf][1] < point_z < self.shelf_height[self.shelf][1]:
                if detections[i]['box'][0] < left_most:
                    left_most = detections[i]['box'][0]
                    left_most_object = i
                if detections[i]['box'][0] > right_most:
                    right_most = detections[i]['box'][0]
                    right_most_object = i
        return left_most_object, right_most_object
    
    def visualiseRVIZ(self, cv_image):
        
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.pub_opencv.publish(ros_image) 
        
        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tostring()
        # Publish new image
        self.pub_compressed_img.publish(msg)

if __name__ == "__main__":
    
    rospy.init_node('category_detection_node', anonymous=True)
    model_name = rospy.get_param('~model_name')

    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')
    
    # VISUAL = True
    # qi_ip = "192.168.50.44"
            
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
 
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    CategoryDetection(model_name, cameras, VISUAL)