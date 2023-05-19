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
from models.PoseDetection.MoveNet_MultiPose.movenet_multipose import MoveNetMultiPose

# import utils
from perception_utils.bcolors import bcolors
import perception_utils.distances_utils as distances_utils
from perception_utils.utils import get_pkg_path
import tf2_ros
import time

#import ros msg and srv
from robobreizh_msgs.msg import Person, PersonList
from robobreizh_msgs.srv import person_features_detection_service

class PersonFeatureDetection():
        
    def __init__(self , yolo_model, face_model, cafffe_age_model, age_gender_model, glass_model, pose_model, colour_csv, cameras: nc.NaoqiCameras, VISUAL) -> None:
        
        self.VISUAL = VISUAL
        self._cameras = cameras
        self.pkg_path = get_pkg_path()
        self.conf_threshold = 0.5
        self.nms_threshold = 0.5
    
        self.yolo_clothes_detector = YOLOV8(model_name=yolo_model,  conf_thres=self.conf_threshold, iou_thres=self.nms_threshold)
        self.face_detector = FaceDetection(face_model_name = face_model)
        self.age_gender_detector = AgeGender(age_gender_model = age_gender_model)
        self.caffe_age_detector = AgePrediction(age_model_name = cafffe_age_model)
        self.glass_detector = GlassDetection(glass_model_name=glass_model) 
        self.colour_detector = ColourDetection(colour_csv_file_name = colour_csv)
        self.pose_detector = MoveNetMultiPose(pose_model_name = pose_model)
    
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
        
        # ori_rgb_image_320 = cv2.resize(ori_rgb_image_640, (320,240), interpolation = cv2.INTER_AREA)
        
        # Person Detection
        detections = self.yolo_detector.inference(ori_rgb_image_320)
                
        if (len(detections) > 0):
            for i in range(len(detections)):
                if (detections[i]['class_name'] == 'person') :
                    person_start_x = round(detections[i]['box'][0])
                    person_end_x = round((detections[i]['box'][0] + detections[i]['box'][2]))
                    person_start_y = round(detections[i]['box'][1])
                    person_end_y = round((detections[i]['box'][1] + detections[i]['box'][3]))
                    
                    cropped_person = ori_rgb_image_320[int(person_start_y):int(person_end_y), int(person_start_x): int(person_end_x)]
                    
                    # cropped person from 640 resolution for face detection
                    # cropped_person_for_face_detection = ori_rgb_image_640[int(person_start_y*2):int(person_end_y*2), int(person_start_x*2): int(person_end_x*2)]

                    # Distance Detection
                    dist, point_x, point_y, point_z, _, _ = distances_utils.detectDistanceResolution(
                            ori_depth_image, person_start_x, person_end_y, person_start_y, person_end_x, resolutionRGB=[ori_rgb_image_320.shape[1], ori_rgb_image_320.shape[0]])
                    
                    if (cropped_person.shape[0] == 0 or cropped_person.shape[0] == 0) :  
                        rospy.loginfo(
                            bcolors.R+"[RoboBreizh - Vision]        No Person Detected within range. "+bcolors.ENDC)                        
                    else:
                        # Detect Clothes Colour
                        clothes_image, clothes_start_x, clothes_start_y, clothes_end_x, clothes_end_y = self.clothesDetection(cropped_person)
                        ok, clothes_color,_, mapping = self.colour_detector.inference(clothes_image, 'clothes')

                        # Face Detection
                        output, cropped_face_image, face_start_x, face_start_y, face_end_x, face_end_y = self.face_detector.inference(cropped_person)
                        
                        # rescale back to 320 for visualisation
                        # face_start_x, face_start_y, face_end_x, face_end_y = int(face_start_x/2), int(face_start_y/2), int(face_end_x/2), int(face_end_y/2)
                        
                        # Glasses Detection
                        if (cropped_face_image is None):
                            age = ""
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
                            
                            # # Glasses detection
                            # glassesOn = self.glass_detector.inference(cropped_face_image)
                            # if glassesOn:
                            #     glasses = "Glasses found"
                            # else:
                            #     glasses = "No Glasses found"
                        
                        time_end = time.time()
        
                        rospy.loginfo("Total time inference: " + str(time_end-time_start))
                        
                        if self.VISUAL:
                            # Display person
                            cv2.rectangle(ori_rgb_image_320, (int(person_start_x), int(person_start_y)) , (int(person_end_x), int(person_end_y)), (255,0,0), 2)
                            # Display face
                            cv2.rectangle(ori_rgb_image_320, (int(face_start_x+person_start_x), int(face_start_y+person_start_y)) , (int(face_end_x+person_start_x), int(face_end_y+person_start_y)), (0,0,255), 2)
                            # Display Clothes
                            cv2.rectangle(ori_rgb_image_320, (int(clothes_start_x+person_start_x), int(clothes_start_y+person_start_y)) , (int(clothes_end_x+person_start_x), int(clothes_end_y+person_start_y)), (255,255,0), 2)

                            self.visualiseRVIZ(ori_rgb_image_320)

                        if dist <= float(distanceMax):
                            # Create Person object
                            person = Person()
                            # Person attributes
                            person.name = String("")
                            person.clothes_color = String(clothes_color)
                            person.age = String(age_caffee)
                            person.gender = String(gender)
                            person.skin_color = String(skin_color)
                            person.glasses = String("")
                            person.height = 0.0
                            person.distance = dist
                            person.coord.x = point_x
                            person.coord.y = point_y
                            person.coord.z = point_z
                            
                            person_list.person_list.append(person)
        else:
            rospy.loginfo(
                bcolors.R+"[RoboBreizh - Vision]        No Objects Detected within range. "+bcolors.ENDC)
                
        return person_list
 
    def visualiseRVIZ(self, image):
        
        ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pub_cv.publish(ros_image) 
    
    def clothesDetection(self, cropped_person):
                        
        # cropped clothes
        cropped_start_x = 0
        cropped_start_y = (0+cropped_person.shape[0])/2
        cropped_end_x = cropped_person.shape[1]
        cropped_end_y = (0+cropped_person.shape[0])/1.2
        clothes_image = cropped_person[int(cropped_start_y):int(cropped_end_y), int(cropped_start_x):int(cropped_end_x)]
                
        return clothes_image, cropped_start_x, cropped_start_y, cropped_end_x, cropped_end_y


if __name__ == "__main__":
    
    rospy.init_node('person_feature_detection_node', anonymous=True)
    VISUAL :bool = rospy.get_param('~visualize')
    qi_ip = rospy.get_param('~qi_ip')
    
    depth_camera_res = res3D.R320x240
    rgb_camera_res = res2D.R320x240
    
    cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    PersonFeatureDetection(yolo_model="clothes_320", 
                           face_model="face_detection_yunet_2022mar.onnx", 
                           cafffe_age_model= "age_net.caffemodel",
                           age_gender_model = "AgeGenderTFlite", 
                           glass_model = "shape_predictor_5_face_landmarks.dat", 
                           pose_model = 'movenet_multipose',
                           colour_csv = "new_colorsV2.csv", 
                           cameras=cameras, VISUAL=VISUAL)