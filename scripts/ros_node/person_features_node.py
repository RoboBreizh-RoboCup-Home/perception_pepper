#!/usr/bin/env python
# import roslib
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import Camera.Naoqi_camera as nc
from Camera.naoqi_camera_types import CameraID, CameraResolution2D as res2D, CameraResolution3D as res3D, ColorSpace2D as cs2D, ColorSpace3D as cs3D
from cv_bridge import CvBridge
import cv2

# import models
from models.ObjectDetection.YOLOV8.yolov8 import YOLOV8
from models.FaceDetection.onnx_facedetection import FaceDetection
from models.AgeGenderDetection.age_gender_detection import AgeGender
from models.AgeGenderDetection.CaffeAgeDetection import AgePrediction
from models.GlassesDetection.GlassesDetection import GlassDetection
from models.ColourDetection.ColourDetection import ColourDetection

# import utils
from perception_utils.bcolors import bcolors
import perception_utils.distances_utils as distances_utils
import perception_utils.transform_utils as tf_utils
from perception_utils.utils import get_pkg_path
import time

#import ros msg and srv
from robobreizh_msgs.msg import Person, PersonList
from robobreizh_msgs.srv import person_features_detection_service

class PersonFeatureDetection():
        
    def __init__(self , yolo_model, face_model, cafffe_age_model, age_gender_model, glass_model, colour_csv, cameras: nc.NaoqiCameras, VISUAL) -> None:
        
        self.VISUAL = VISUAL        
        self._cameras = cameras
        self.pkg_path = get_pkg_path()
        self.conf_threshold = 0.5
        self.nms_threshold = 0.5
    
        self.yolo_clothes_detector = YOLOV8(model_name=yolo_model,  _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        self.face_detector = FaceDetection(face_model_name = face_model)
        self.age_gender_detector = AgeGender(age_gender_model = age_gender_model)
        self.caffe_age_detector = AgePrediction(age_model_name = cafffe_age_model)
        self.glass_detector = GlassDetection(glass_model_name=glass_model) 
        self.colour_detector = ColourDetection(colour_csv_file_name = colour_csv, color_type="hue")
    
        if self.VISUAL: 
            self.bridge = CvBridge()
            self.pub_cv = rospy.Publisher('/roboBreizh_detector/person_feature_detection_image', Image, queue_size=10)
            
        self.initPersonDescriptionService()
 
    def initPersonDescriptionService(self):
        rospy.Service('/robobreizh/perception_pepper/person_feature_detection',
                        person_features_detection_service, self.handle_ServicePerceptionHuman)
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Person Features Detection. "+bcolors.ENDC)
        rospy.spin()

    def handle_ServicePerceptionHuman(self, person_features_detection_service):

        # entries list of maximum distance for person detection
        distanceMax = person_features_detection_service.entries_list.distanceMaximum
                
        # Create Person List to store attributes for each person
        person_list= PersonList()
        person_list.person_list = []
        
        time_start = time.time()
        # retrieve rgb and depth image from Naoqi camera
        ori_rgb_image_320, ori_depth_image = self._cameras.get_image(out_format="cv2")
                
        # Clothes Detection
        detections = self.yolo_clothes_detector.inference(ori_rgb_image_320)

        if (len(detections) > 0):
                clothes_start_x = round(detections[0]['box'][0])
                clothes_end_x = round((detections[0]['box'][0] + detections[0]['box'][2]))
                clothes_start_y = round(detections[0]['box'][1])
                clothes_end_y = round((detections[0]['box'][1] + detections[0]['box'][3]))
                
                # Distance Detection
                dist, point_x, point_y, point_z, _, _ = distances_utils.detectDistanceResolution(
                        ori_depth_image, clothes_start_x, clothes_end_y, clothes_start_y, clothes_end_x, resolutionRGB=[ori_rgb_image_320.shape[1], ori_rgb_image_320.shape[0]])
                
                odom_point = tf_utils.compute_absolute_pose([point_x,point_y,point_z])

                
                if dist <= float(distanceMax):
                    
                    clothes_style = str(detections[0]['class_name'])
                    cropped_clothes = ori_rgb_image_320[int(clothes_start_y):int(clothes_end_y), int(clothes_start_x): int(clothes_end_x)]
                    
                    ok, clothes_color,_, mapping = self.colour_detector.inference(cropped_clothes, 'clothes')      
                    
                    # Face Detection
                    output, cropped_face_image, face_start_x, face_start_y, face_end_x, face_end_y = self.face_detector.inference(ori_rgb_image_320)
                    
                    if (cropped_face_image is None):
                        age_caffee = ""
                        gender = ""
                        skin_color = ""
                        rospy.loginfo(
                            bcolors.R+"[RoboBreizh - Vision]    Face not Detected"+bcolors.ENDC)    
                    else:
                        # Face, Age, Gender
                        gender  = self.age_gender_detector.inference(cropped_face_image)
                        
                        # Age from Caffee
                        age_caffee = self.caffe_age_detector.inference(cropped_face_image)
                        
                        # Face Skin Colour detection
                        ok, skin_color,_, mapping = self.colour_detector.inference(cropped_face_image, 'skin')
                        
                    # Create Person object
                    person = Person()
                    # Person attributes
                    person.clothes_color = String(clothes_color)
                    person.clothes_style = String(clothes_style)
                    person.age = String(age_caffee)
                    person.gender = String(gender)
                    person.skin_color = String(skin_color)
                    person.distance = dist
                    person.coord.x = odom_point.x
                    person.coord.y = odom_point.y
                    person.coord.z = odom_point.z
                    
                    person_list.person_list.append(person)
                    
                    if self.VISUAL:
                        # Display clothes
                        cv2.rectangle(ori_rgb_image_320, (int(clothes_start_x), int(clothes_start_y)) , (int(clothes_end_x), int(clothes_end_y)), (255,0,0), 2)
                        # Display face
                        cv2.rectangle(ori_rgb_image_320, (int(face_start_x), int(face_start_y)) , (int(face_end_x), int(face_end_y)), (0,0,255), 2)
                    
                else:
                    rospy.loginfo(
                        bcolors.R+"[RoboBreizh - Vision]    Clothes/Person Detected but Out of Range"+bcolors.ENDC)                                  
                  
        else:
            rospy.loginfo(
                bcolors.R+"[RoboBreizh - Vision]    Clothes/Person not Detected"+bcolors.ENDC)              
                
        time_end = time.time()

        rospy.loginfo("Total time inference: " + str(time_end-time_start))
        
        if self.VISUAL:
            self.visualiseRVIZ(ori_rgb_image_320)

        return person_list
 
    def visualiseRVIZ(self, image):
        
        ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pub_cv.publish(ros_image) 


if __name__ == "__main__":
    
    
    rospy.init_node('person_feature_detection_node', anonymous=True)

    VISUAL = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')
    
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240

    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    PersonFeatureDetection(yolo_model="clothes_320", 
                           face_model="face_detection_yunet_2022mar.onnx", 
                           cafffe_age_model= "age_net.caffemodel",
                           age_gender_model = "AgeGenderTFlite", 
                           glass_model = "shape_predictor_5_face_landmarks.dat", 
                           colour_csv = "new_colorsV2.csv", 
                           cameras=cameras, VISUAL=VISUAL)