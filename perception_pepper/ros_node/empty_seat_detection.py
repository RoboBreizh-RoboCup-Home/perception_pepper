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
from models.ObjectDetection.SSDInception_coco2017.ssd_inception_coco import SSDInception
from perception_utils.bcolors import bcolors
import perception_utils.distances_utils as distances_utils
import perception_utils.display_utils as display_utils
import perception_utils.transform_utils as tf_utils
from perception_utils.objects_detection_utils import *
from perception_utils.utils import get_pkg_path
from robobreizh_msgs.msg import *
from robobreizh_msgs.srv import *
from visualization_msgs.msg import MarkerArray
import tf2_ros
import time

class EmptySeatDetection():

    def __init__(self, cameras: nc.NaoqiCameras, VISUAL):

        self.VISUAL = VISUAL
        self._cameras = cameras
        self.conf_threshold = 0.4
        self.chairs_iou = 0.28
        self.pkg_path = get_pkg_path()
        self.objectRequested_list = []
        self.distanceMax = 0
        self.ssd_coco_detector = SSDInception(_conf_threshold= self.conf_threshold)
        self.coco_class_list = self.ssd_coco_detector.coco_classes_list

        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]    Loading Object Detection weights done"+bcolors.ENDC)

        if self.VISUAL:
            self.bridge = CvBridge()
            self.pub_opencv = rospy.Publisher('/roboBreizh_detector/empty_seat_detection', Image, queue_size=10)
            self.pub_compressed_img = rospy.Publisher("/roboBreizh_detector/empty_seat_compressed_image",
            CompressedImage,  queue_size=10)
            self.marker_arr_pub = rospy.Publisher(
                '/roboBreizh_detector/visualization_marker_array', MarkerArray, queue_size=1)

        self.initObjectDescriptionService()

    def initObjectDescriptionService(self):
        rospy.Service('/robobreizh/perception_pepper/seat_detection_service',
                        seat_detection_service, self.handle_ServicePerceptionObject)
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Empty Seat Detection. "+bcolors.ENDC)
        rospy.spin()

    def handle_ServicePerceptionObject(self,seat_detection_service):

        objects_Requested = seat_detection_service.entries_list

        self.distanceMax = seat_detection_service.entries_list.distanceMaximum

        for i in range(len(objects_Requested.obj)):
            self.objectRequested_list.append(objects_Requested.obj[i].data)

        print("Object Requested List: ")
        print(self.objectRequested_list)

        print("Distance Maximum: ")
        print(self.distanceMax)

        t_start_computing = time.time()

        print("     SEAT info : ")

        seat_list = ObjectList()
        seat_list.object_list = []

        arr_empty_chairs = []

        ori_rgb_image_320, ori_depth_image = self._cameras.get_image(out_format="cv2")

        outputs_coco = self.ssd_coco_detector.inference(ori_rgb_image_320, self.objectRequested_list)

        ori_rgb_image_320, arr_persons, arr_empty_chairs, arr_taken_chairs = has_chairs_coco_couch(ori_rgb_image_320,
                self.coco_class_list, outputs_coco, ori_rgb_image_320.shape[1], ori_rgb_image_320.shape[0])

        if len(arr_empty_chairs)>1:
            for i in range(len(arr_empty_chairs)):
                for j in range(i+1, len(arr_empty_chairs)):
                    iou = (intersection_over_union(arr_empty_chairs[i].xyxy, arr_empty_chairs[j].xyxy))
                    if iou > self.chairs_iou:
                        del arr_empty_chairs[i]

        # ------------------ TIMING  ------------------------------
        t_end_computing = time.time()
        print(bcolors.B+"     --> seat detection inference time " +
                str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

        for chair in arr_empty_chairs:
            if ori_depth_image.size == 0:
                print(bcolors.WARNING +
                        '     depth_image image  EMPTY'+bcolors.ENDC)
            else:
                # chair.xyxy = [top, left, bottom, right]
                top = chair.xyxy[0]
                left = chair.xyxy[1]
                bottom = chair.xyxy[2]
                right = chair.xyxy[3]
                dist, point_x, point_y, point_z, Xcenter, Ycenter = distances_utils.detectDistanceResolution(
                    ori_depth_image, left, bottom, top, right, resolutionRGB=[ori_rgb_image_320.shape[1], ori_rgb_image_320.shape[0]])

                odom_point = tf_utils.compute_absolute_pose([point_x,point_y,point_z])

                if dist <= self.distanceMax:
                    ############## store for MANAGER ##############
                    obj = Object()
                    obj.label.data = "seat"
                    obj.coord.x = odom_point.x
                    obj.coord.y = odom_point.y
                    obj.coord.z = odom_point.z
                    obj.distance = float(dist)
                    obj.height_img = bottom-top  # for Rviz scaling
                    obj.width_img = right-left
                    seat_list.object_list.append(obj)

                    if self.VISUAL:
                        display_utils.show_RViz_marker_arr(self.marker_arr_pub, seat_list, DURATION=50)

                else:
                    rospy.loginfo(
                        bcolors.R+"[RoboBreizh - Vision]        Seats Detected but not within range. "+bcolors.ENDC)
        if self.VISUAL:
            self.visualiseRVIZ(ori_rgb_image_320)


        return seat_list

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

    rospy.init_node('seat_detection_node', anonymous=True)

    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')

    # VISUAL = True
    # qi_ip = "192.168.50.44"

    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240

    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    EmptySeatDetection(cameras, VISUAL)
