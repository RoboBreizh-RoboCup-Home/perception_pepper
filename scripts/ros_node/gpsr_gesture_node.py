#!/usr/bin/env python

# import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
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
from models.PoseDetection.MoveNet_MultiPose.movenet_multipose import MoveNetMultiPose
from models.PoseDetection.roboBreizh_Utils_pose import visualize
from models.PoseDetection.roboBreizh_Utils_pose import is_pointing
from models.PoseDetection.roboBreizh_Utils_pose import is_waving_gpsr
from robobreizh_msgs.msg import *
from robobreizh_msgs.srv import *
import tf2_ros
import time

class GPSRGestureDetection():
        
    def __init__(self , model_name, cameras: nc.NaoqiCameras, VISUAL) -> None:
        
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.model_name = model_name
        self.person_threshold = 0.3
        self.distanceMax = 0
        self.pkg_path = get_pkg_path()
        self.pose_model = MoveNetMultiPose(pose_model_name=model_name)
        
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]    Loading Multipose Detection weights done"+bcolors.ENDC)
        
        if self.VISUAL: 
            self.bridge = CvBridge()
            self.pub_opencv = rospy.Publisher('/roboBreizh_detector/gpsr_gesture_detection_image', Image, queue_size=10)
        
        self.initHandPointingDescriptionService()
        
    def initHandPointingDescriptionService(self):
        rospy.Service('/robobreizh/perception_pepper/gpsr_gesture_detection',
                        gpsr_gesture_detection, self.handle_ServiceGPSRGesture)
            
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting GPSR Gesture Detection. "+bcolors.ENDC)
        rospy.spin()

    def handle_ServiceGPSRGesture(self, gpsr_gesture_detection):
 
        person_list = PersonPoseList()
        person_list.person_pose_list =[]
               
        rightList = []
        topList = []
        
        iswaving = False 
        raising_left = False 
        raising_right = False 
        pointing_right = False 
        pointing_left = False
        
        self.distanceMax = gpsr_gesture_detection.distance_max
        
        rgb_image, depth_image = self._cameras.get_image(out_format="cv2")
        
        list_person = self.pose_model.inference(rgb_image)
        
        if len(list_person)>0:
            for person in list_person:
                if person.score < self.person_threshold:
                    pass
                else:
                    start_x = person.bounding_box.start_point.x
                    start_y = person.bounding_box.start_point.y
                    end_x = person.bounding_box.end_point.x
                    end_y = person.bounding_box.end_point.y
                                    
                    dist, point_x, point_y, point_z, _, _ = distances_utils.detectDistanceResolution(
                        depth_image, start_x, end_y, start_y, end_x, resolutionRGB=[rgb_image.shape[1], rgb_image.shape[0]])
                    
                    # change point from camera frame to odom, see the function for more details
                    odom_point = tf_utils.compute_absolute_pose([point_x,point_y,point_z])
                    
                    mid_y = (start_y+end_y)/2
                    mid_x = (start_x+end_x)/2
                    
                    bPointing, bRight, bTop = is_pointing(person)
                    
                    if bRight == 1:
                        rospy.loginfo("Human Pointing to Right Found")
                        pointing_right = True
                    else:
                        rospy.loginfo("Human Pointing to Left Found")
                        pointing_left = True
                    
                    waving_side = is_waving_gpsr(person)
                    
                    if waving_side == "waving_left":
                        rospy.loginfo("Human Waving Found")
                        rospy.loginfo("Human Raising Left Arm Found")
                        raising_left = True
                        iswaving = True
                    if waving_side == "waving_right":
                        rospy.loginfo("Human Waving Found")
                        rospy.loginfo("Human Raising Right Arm Found")
                        raising_right = True
                        iswaving = True
                        
                    if dist <= self.distanceMax:
                        if bPointing:
                            rightList.append(bRight)
                            topList.append(bTop)
                            
                        person_pose = PersonPose()
                        person_pose.person_id = String(str(person.id))
                        person_pose.waving = iswaving
                        person_pose.raising_left = raising_left
                        person_pose.raising_right = raising_right
                        person_pose.pointing_left = pointing_left
                        person_pose.pointing_right = pointing_right

                        person_pose.distance = dist
                        person_pose.coord.x = odom_point.x
                        person_pose.coord.y = odom_point.y
                        person_pose.coord.z = odom_point.z
                        person_list.person_pose_list.append(person_pose)
                    
                        if self.VISUAL:     
                            rgb_image = visualize(rgb_image, list_person)
                            cv2.circle(rgb_image, (int(mid_x), int(mid_y)), 1, (255,0,0), 1 )
                    else:
                        rospy.loginfo(bcolors.R+"[RoboBreizh - Pose/GPSR Gesture Detection] Person Detected but not within Range..."+bcolors.ENDC)
                
                rospy.loginfo(bcolors.ON_PURPLE+"[RoboBreizh - Pose/GPSR Gesture Detection] detection done..."+bcolors.ENDC)
        else:
            rospy.loginfo(bcolors.R+"[RoboBreizh - Pose/GPSR Gesture Detection] No Person Detected"+bcolors.ENDC)
        
        if self.VISUAL:
            self.visualiseRVIZ(rgb_image)
        
        return person_list
    
    def visualiseRVIZ(self, chair_image):
        
        cv_chair_image = self.bridge.cv2_to_imgmsg(chair_image, "bgr8")
        self.pub_opencv.publish(cv_chair_image) 

if __name__ == "__main__":
    
    rospy.init_node('gpsr_gesture_node', anonymous=True)

    # VISUAL = rospy.get_param('~visualize')
    # qi_ip = rospy.get_param('~qi_ip')
    
    VISUAL = True
    qi_ip = "192.168.50.44"
            
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
 
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    GPSRGestureDetection("movenet_multipose", cameras, VISUAL)
