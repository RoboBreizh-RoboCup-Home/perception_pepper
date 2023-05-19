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
from models.ObjectDetection.YOLOV8.yolov8 import YOLOV8
from robobreizh_msgs.msg import *
from robobreizh_msgs.srv import *
import tf2_ros
import time

class ChairDetection():
        
    def __init__(self , model_name, cameras: nc.NaoqiCameras, VISUAL) -> None:
        
        #self.VISUAL :bool = rospy.get_param('~visualize')
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.model_name = model_name
        self.conf_threshold = 0.3
        self.nms_threshold = 0.4
        self.pkg_path = get_pkg_path()
        self.yolo_seat_detector = YOLOV8(model_name=self.model_name, _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]    Loading Person Detection weights done"+bcolors.ENDC)
        
        if self.VISUAL: 
            self.bridge = CvBridge()
            self.pub_opencv = rospy.Publisher('/roboBreizh_detector/person_detection_image', Image, queue_size=10)
        
        self.initChairDescriptionService()
        
    def initChairDescriptionService(self):
        rospy.Service('/robobreizh/perception_pepper/chair_detection',
                        seat_detection_service, self.handle_ServicePerceptionHuman)
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Chair Detection. "+bcolors.ENDC)
        rospy.spin()

    def handle_ServicePerceptionHuman(self, seat_detection_service):
        

        
        time_start = time.time()
        
        # retrieve rgb and depth image from Naoqi camera
        ori_rgb_image, ori_depth_image = self._cameras.get_image(out_format="cv2")

        detections = self.yolo_seat_detector.inference(ori_rgb_image)
        
        if (len(detections)>0):
            
            chair_bounding_box = []
            couch_bounding_box = []
            person_bounding_box  = [] 
            chair_count = 0
            couch_count = 0
            person_count = 0
            chair_count_list = []
            couch_count_list = []
            person_count_list = []
            
            for i in range(len(detections)):
                if (detections[i]['class_name'] == 'chair'):
                    chair_count+=1
                    object_name = detections[i]['class_name']
                    chair_start_x = round(detections[i]['box'][0])
                    chair_end_x = round((detections[i]['box'][0] + detections[i]['box'][2]))
                    chair_start_y = round(detections[i]['box'][1])
                    chair_end_y = round((detections[i]['box'][1] + detections[i]['box'][3]))
                    
                    chair_bounding_box = [chair_start_x, chair_start_y, chair_end_x,chair_end_y]
                                        
                    # Distance Detection
                    chair_dist, chair_point_x, chair_point_y, chair_point_z, _, _ = distances_utils.detectDistanceResolution(
                            ori_depth_image, chair_start_x, chair_end_y, chair_start_y, chair_end_x, resolutionRGB=[ori_rgb_image.shape[1], ori_rgb_image.shape[0]])
                    
                    chair_odom_point = tf_utils.compute_absolute_pose([chair_point_x,chair_point_y,chair_point_z])

                    seatDict = {}
                    chair_bounding_box_dict = {"chair_bounding_box" : chair_bounding_box}
                    seatDict[str(object_name + str(chair_count))] = (chair_bounding_box_dict)
                    odom_point_dict = {"odom point" : chair_odom_point}
                    seatDict[str(object_name + str(chair_count))].update(odom_point_dict)
                    dist_dict = {"distance" : chair_dist}
                    seatDict[str(object_name + str(chair_count))].update(dist_dict)
                    
                    if self.VISUAL:
                        cv2.rectangle(ori_rgb_image, (chair_start_x, chair_start_y), (chair_end_x,chair_end_y), (255,255,0), 0)
                        
                    chair_count_list.append(seatDict)
                    
                if (detections[i]['class_name'] == 'couch'):
                    couch_count+=1
                    object_name = detections[i]['class_name']
                    couch_start_x = round(detections[i]['box'][0])
                    couch_end_x = round((detections[i]['box'][0] + detections[i]['box'][2]))
                    couch_start_y = round(detections[i]['box'][1])
                    couch_end_y = round((detections[i]['box'][1] + detections[i]['box'][3]))
                    
                    couch_bounding_box = [couch_start_x, couch_start_y, couch_end_x, couch_end_y]
                                        
                    # Distance Detection
                    couch_dist, couch_point_x, couch_point_y, couch_point_z, _, _ = distances_utils.detectDistanceResolution(
                            ori_depth_image, couch_start_x, couch_end_y, couch_start_y, couch_end_x, resolutionRGB=[ori_rgb_image.shape[1], ori_rgb_image.shape[0]])
                    
                    couch_odom_point = tf_utils.compute_absolute_pose([couch_point_x,couch_point_y,couch_point_z])

                    couchDict = {}
                    couch_bounding_box_dict = {"couch_bounding_box" : couch_bounding_box}
                    couchDict[str(object_name + str(couch_count))] = (couch_bounding_box_dict)
                    odom_point_dict = {"odom point" : couch_odom_point}
                    couchDict[str(object_name + str(couch_count))].update(odom_point_dict)
                    couch_dist_dict = {"distance" : couch_dist}
                    couchDict[str(object_name + str(couch_count))].update(couch_dist_dict)
                    
                    if self.VISUAL:
                        cv2.rectangle(ori_rgb_image, (couch_start_x, couch_start_y), (couch_end_x,couch_end_y), (255,255,0), 0)
                        
                    couch_count_list.append(couchDict)

                if (detections[i]['class_name'] == 'person') :
                    
                    person_count+=1
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

                    personDict = {}
                    person_bounding_box_dict = {"person_bounding_box" : person_bounding_box}
                    personDict[str(person_name + str(person_count))] = (person_bounding_box_dict)
                    odom_point_dict = {"odom point" : person_odom_point}
                    personDict[str(person_name + str(person_count))].update(odom_point_dict)
                    dist_dict = {"distance" : person_dist}
                    personDict[str(person_name + str(person_count))].update(dist_dict)
                    
                    person_count_list.append(personDict)

                    if self.VISUAL:
                        cv2.rectangle(ori_rgb_image, (person_start_x, person_start_y), (person_end_x,person_end_y), (255,0,0), 0)
            
            if len(chair_count_list) > 0:
                chair_status_dict = {}
                for chair in chair_count_list:
                    seat_occupied_list = []
                    seatOccupied = False
                    if len(person_count_list) > 0:
                        for person in person_count_list:
                            iou = intersection_over_union(person[list(person.keys())[0]]['person_bounding_box'], chair[list(chair.keys())[0]]['chair_bounding_box'])
                            distance = seat_person_mid_point_distance(person[list(person.keys())[0]]['person_bounding_box'], chair[list(chair.keys())[0]]['chair_bounding_box'])
                            print(distance)
                            if iou >= self.iou_threshold_seat and distance <=self.mid_point_seat_person_margin:
                                seatOccupied = True
                                seat_occupied_list.append(seatOccupied)
                            else:
                                seat_occupied_list.append(seatOccupied)
                    else:
                        seat_occupied_list.append(seatOccupied)
                        
                    chair_status_dict[str(list(chair.keys())[0])] = seat_occupied_list
                                    
                for chair_name, occupied_result in chair_status_dict.items():
                    rospy.loginfo(
                        bcolors.CYAN+"[RoboBreizh - Vision]        " + chair_name + " Detected. "+bcolors.ENDC)      
                    if True in occupied_result:
                        for chair in chair_count_list:
                            if str(list(chair.keys())[0]) == chair_name :
                                chair_obj = Object()
                                # Chair attributes
                                chair_obj.label = String(chair_name)
                                chair_obj.distance = chair[chair_name]['distance']
                                chair_obj.coord.x = chair[chair_name]['odom point'].x
                                chair_obj.coord.y = chair[chair_name]['odom point'].y
                                chair_obj.coord.z = chair[chair_name]['odom point'].z   
                                chair_obj.status = String("Occupied")
                                seat_list.object_list.append(chair_obj)
                    else:
                        for chair in chair_count_list:
                            if str(list(chair.keys())[0]) == chair_name :
                                chair_obj = Object()
                                # Chair attributes
                                chair_obj.label = String(chair_name)
                                chair_obj.distance = chair[chair_name]['distance']
                                chair_obj.coord.x = chair[chair_name]['odom point'].x
                                chair_obj.coord.y = chair[chair_name]['odom point'].y
                                chair_obj.coord.z = chair[chair_name]['odom point'].z   
                                chair_obj.status = String("Empty")  
                                seat_list.object_list.append(chair_obj)  
                                
            else:
                rospy.loginfo(
                    bcolors.R+"[RoboBreizh - Vision]        No Chairs Detected. "+bcolors.ENDC) 
                
            if len(couch_count_list)>0:
                couch_status_dict = {}
                for couch in couch_count_list:
                    seat_occupied_list = []
                    seatOccupied = False
                    if len(person_count_list) > 0:
                        for person in person_count_list:
                            iou = intersection_over_union(person[list(person.keys())[0]]['person_bounding_box'], couch[list(couch.keys())[0]]['couch_bounding_box'])
                            distance = seat_person_mid_point_distance(person[list(person.keys())[0]]['person_bounding_box'], couch[list(couch.keys())[0]]['couch_bounding_box'])
                            print(distance)
                            if iou >= self.iou_threshold_seat and distance <=self.mid_point_seat_person_margin:
                                seatOccupied = True
                                seat_occupied_list.append(seatOccupied)
                            else:
                                seat_occupied_list.append(seatOccupied)
                    else:
                        seat_occupied_list.append(seatOccupied)
                        
                    couch_status_dict[str(list(couch.keys())[0])] = seat_occupied_list
                                    
                for couch_name, occupied_result in couch_status_dict.items():
                    rospy.loginfo(
                        bcolors.CYAN+"[RoboBreizh - Vision]        " + couch_name + " Detected. "+bcolors.ENDC)                        
                    if True in occupied_result:
                        for couch in couch_count_list:
                            if str(list(couch.keys())[0]) == couch_name :
                                couch_obj = Object()
                                # Couch attributes
                                couch_obj.label = String(couch_name)
                                couch_obj.distance = couch[couch_name]['distance']
                                couch_obj.coord.x = couch[couch_name]['odom point'].x
                                couch_obj.coord.y = couch[couch_name]['odom point'].y
                                couch_obj.coord.z = couch[couch_name]['odom point'].z   
                                couch_obj.status = String("Occupied")
                                seat_list.object_list.append(couch_obj)
                    else:
                        for couch in couch_count_list:
                            if str(list(couch.keys())[0]) == couch_name :
                                couch_obj = Object()
                                # Couch attributes
                                couch_obj.label = String(couch_name)
                                couch_obj.distance = couch[couch_name]['distance']
                                couch_obj.coord.x = couch[couch_name]['odom point'].x
                                couch_obj.coord.y = couch[couch_name]['odom point'].y
                                couch_obj.coord.z = couch[couch_name]['odom point'].z   
                                couch_obj.status = String("Empty")  
                                seat_list.object_list.append(couch_obj)              
                                
            else:
                rospy.loginfo(
                    bcolors.R+"[RoboBreizh - Vision]        No Couch Detected. "+bcolors.ENDC)    
                                
        else:
            rospy.loginfo(
                bcolors.R+"[RoboBreizh - Vision]        No Objects Detected. "+bcolors.ENDC)               
                        
        if self.VISUAL: 
            self.visualiseRVIZ(ori_rgb_image)


        time_end = time.time()
        rospy.loginfo("Total time inference: " + str(time_end-time_start))
        
        return seat_list 
    
    def visualiseRVIZ(self, chair_image):
        
        cv_chair_image = self.bridge.cv2_to_imgmsg(chair_image, "bgr8")
        self.pub_opencv.publish(cv_chair_image) 

if __name__ == "__main__":
    
    rospy.init_node('person_detection_node', anonymous=True)

    # VISUAL = rospy.get_param('~visualize')
    # qi_ip = rospy.get_param('~qi_ip')
    
    VISUAL = True
    
    qi_ip = "192.168.50.44"
    
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
 
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    ChairDetection("receptionist_320", cameras, VISUAL)