#!/usr/bin/env python

# import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
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
from models.PoseDetection.roboBreizh_Utils_pose import is_waving
from robobreizh_msgs.msg import *
from robobreizh_msgs.srv import *
import tf2_ros
import time

class HandWavingDetection():
        
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
            self.pub_opencv = rospy.Publisher('/roboBreizh_detector/hand_waving_image', Image, queue_size=10)
            self.pub_compressed_img = rospy.Publisher("/roboBreizh_detector/hand_waving_compressed_image",
            CompressedImage,  queue_size=10)
        
        self.initHandWavingDescriptionService()
        
    def initHandWavingDescriptionService(self):
        rospy.Service('/robobreizh/perception_pepper/hand_waving',
                        hand_waving_detection, self.handle_ServicePerceptionObject)
            
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Hand Waving Detection. "+bcolors.ENDC)
        rospy.spin()

    def handle_ServicePerceptionObject(self, hand_waving_detection):
        
        person_list = PersonPoseList()
        person_list.person_pose_list =[]
        
        self.distanceMax = hand_waving_detection.entries_list.distanceMaximum
        
        rgb_image, depth_image = self._cameras.get_image(out_format="cv2")
        
        time_start = time.time()
        
        list_person = self.pose_model.inference(rgb_image)
        
        time_end = time.time()
        
        # print(time_end - time_start)
        
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
                    
                    iswaving = is_waving(person)
                    
                    if dist <= self.distanceMax:
                        if iswaving:
                            person_pose = PersonPose()
                            person_pose.person_id = String(str(person.id))
                            person_pose.waving = iswaving
                            person_pose.distance = dist
                            person_pose.coord.x = odom_point.x
                            person_pose.coord.y = odom_point.y
                            person_pose.coord.z = odom_point.z
                            person_list.person_pose_list.append(person_pose)
                        if self.VISUAL:     
                            rgb_image = visualize(rgb_image, list_person)
                            cv2.circle(rgb_image, (int(mid_x), int(mid_y)), 1, (255,0,0), 1 )
                    else:
                        rospy.loginfo(bcolors.R+"[RoboBreizh - Pose/Hand Waving] Person Detected but not within Range..."+bcolors.ENDC)
                
                rospy.loginfo(bcolors.ON_PURPLE+"[RoboBreizh - Pose/Hand Waving] detection done..."+bcolors.ENDC)
        else:
            rospy.loginfo(bcolors.R+"[RoboBreizh - Pose/Hand Waving] No Person Detected"+bcolors.ENDC)
        
        if self.VISUAL:
            self.visualiseRVIZ(rgb_image)

        return person_list
    
    def visualiseRVIZ(self, chair_image):
        
        cv_chair_image = self.bridge.cv2_to_imgmsg(chair_image, "bgr8")
        self.pub_opencv.publish(cv_chair_image) 
    
        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tostring()
        # Publish new image
        self.pub_compressed_img.publish(msg)

if __name__ == "__main__":
    
    rospy.init_node('hand_waving_node', anonymous=True)

    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')
    
    # VISUAL = True
    # qi_ip = "192.168.50.44"
            
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
 
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    HandWavingDetection("movenet_multipose", cameras, VISUAL)
