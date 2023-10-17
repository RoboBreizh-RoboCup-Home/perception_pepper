#!/usr/bin/env python

# import types
from typing import List
import numpy as np 

# import roslib
import rospy
import actionlib

# import robobreizh msgs
import Camera.Naoqi_camera as nc
from Camera.naoqi_camera_types import CameraResolution2D as res2D, CameraResolution3D as res3D 
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import Point32

from robobreizh_msgs.msg import ObjectList,TrackerAction,TrackerFeedback,TrackerResult, Object
from robobreizh_msgs.srv import *

# RViz Visualization using Marker
from visualization_msgs.msg import MarkerArray
# import utils
import perception_utils.transform_utils as tf_utils
from perception_utils.bcolors import bcolors
import perception_utils.distances_utils as distances_utils
import perception_utils.display_utils as display_utils

from PoseDetection.roboBreizh_Utils_pose import visualize

# import models
from PoseDetection.MoveNet_lightning.movenet_lightning import Movenet

import time

class PersonTrackerNode():
    def __init__(self, model_name , cameras: nc.NaoqiCameras, width, height, VISUAL) -> None:
        
        
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.width = width
        self.height = height
        self.pose_model = Movenet(pose_model_name=model_name, width=self.width, height=self.height)
        self._distance_max = 2.5
        self.person_threshold = 0.4
        if self.VISUAL: 
            self.pub_cv = rospy.Publisher('/roboBreizh_detector/person_tracker_image', Image, queue_size=10)
            self.pub_compressed_img = rospy.Publisher("/roboBreizh_detector/person_tracker_compressed_image",
            CompressedImage,  queue_size=10)
            self.bridge = CvBridge()
             # Publisher for Markers
            self.marker_arr_pub = rospy.Publisher(
                '/roboBreizh_detector/visualization_marker_array', MarkerArray, queue_size=1)

        self._action_server = actionlib.SimpleActionServer(
            'robobreizh/perception_pepper/person_tracker', TrackerAction, self.person_tracker_callback, False)
        self._action_server.start()
        
        rospy.loginfo("Starting Action Server")

    def person_tracker_callback(self, goal_handle):
        """
        Callback function for the action server.
        """
        
        rospy.loginfo("Action Server (Tracker) Callback")
        # Initiate the feedback message's current_num as the action request's starting_num
        feedback_msg = TrackerFeedback()
        success = goal_handle.startActionServer
        
        while success:
            time_start = time.time()
            
            rgb_image, depth_image = self._cameras.get_image(out_format="cv2")
            
            list_person = [self.pose_model.inference(rgb_image)]
            
            person_info_list = []
            
            gesture = ""
            
            for person in list_person:
                if person.score < self.person_threshold:
                    pass
                else:
                    # rospy.loginfo(person.score)
                    # rospy.loginfo("     -->  Person detected")
                    
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

                    scaled_mid_x = mid_x / self.width
                    
                    person_info_list.append((person.id, dist, odom_point, gesture, scaled_mid_x))
                    
                    if dist > self._distance_max:
                        continue
                    
                    if self.VISUAL:     
                        rgb_image = visualize(rgb_image, list_person)
                        cv2.circle(rgb_image, (int(mid_x), int(mid_y)), 1, (255,0,0), 1 )
                        
                
            rospy.loginfo(bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] detection done..."+bcolors.ENDC)
                        

            if self._action_server.is_preempt_requested():
                self._action_server.set_preempted()
                rospy.loginfo(f'Preempted : {self._action_server.is_preempt_requested()}')
                success = False
                break
                    
            if (len(person_info_list) > 0):
                for i in range(len(person_info_list)):
                    feedback_msg.header.stamp = rospy.Time.now()
                    feedback_msg.header.frame_id = "odom"
                    feedback_msg.person_id = person_info_list[i][0]
                    feedback_msg.dist = person_info_list[i][1]
                    feedback_msg.point = person_info_list[i][2]
                    feedback_msg.gesture.posture.data = person_info_list[i][3]
                    feedback_msg.horizontal_point = float(person_info_list[i][4])

                    if self.VISUAL:  
                        pose_image = self.bridge.cv2_to_imgmsg(rgb_image, "bgr8")
                        self.pub_cv.publish(pose_image)    
                        
                        #### Create CompressedIamge ####
                        msg = CompressedImage()
                        msg.header.stamp = rospy.Time.now()
                        msg.format = "jpeg"
                        msg.data = np.array(cv2.imencode('.jpg', rgb_image)[1]).tostring()
                        # Publish new image
                        self.pub_compressed_img.publish(msg)
                        
                        obj_list = ObjectList()
                        obj_list.object_list =[]
                        obj = Object()
                        obj.label.data = "Person"
                        obj.coord = feedback_msg.point
                        obj.distance = feedback_msg.dist
                        obj.width_img = 320
                        obj.height_img = 240
                        obj_list.object_list.append(obj)
                        display_utils.show_RViz_marker_arr(self.marker_arr_pub, obj_list, DURATION=50)
                        

                    self._action_server.publish_feedback(feedback_msg)

                    # Print log messages
                    rospy.loginfo(f'Someone was detected, publishing feedback')
                    rospy.logdebug(f'Feedback : {feedback_msg}')

                time_end = time.time()
            else:
                feedback_msg.person_id = -1
                feedback_msg.dist = 0
                feedback_msg.point = Point32()
                feedback_msg.gesture.posture.data = gesture
                feedback_msg.horizontal_point = 0.0

                rospy.loginfo(f'Nobody was detected, publishing empty feedback')
                self._action_server.publish_feedback(feedback_msg)
                
                time_end = time.time()
            rospy.loginfo(f"Overall Inference time: {time_end - time_start}")


        if (success == False):
            rospy.loginfo('Shut down Action Server (Tracker)!')
            result = TrackerResult()
            result.is_finished = True
            # Indicate that the goal was successful
            self._action_server.set_succeeded(result)

if __name__ == "__main__":

    rospy.init_node('person_tracker', anonymous=True, log_level=rospy.INFO)
    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')

    # VISUAL = True
    # qi_ip = "192.168.50.44"

    # Cameras setting
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R640x480
    rgb_camera_height = 480
    rgb_camera_width = 640
    
    # Default model: single person tracker 
    pose_model = "movenet_lightning"
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    PersonTrackerNode(pose_model, cameras, rgb_camera_width,rgb_camera_height, VISUAL)
    rospy.spin()
