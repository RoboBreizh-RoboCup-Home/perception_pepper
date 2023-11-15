#!/usr/bin/env python
# import roslib
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from perception_pepper.Camera import NaoqiCameras
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
import time

class PersonFeatureDetection(Node):
        
    def __init__(self , yolo_model, face_model, cafffe_age_model, age_gender_model, glass_model, colour_csv, cameras: NaoqiCameras, VISUAL) -> None:
        super().__init__('FeaturesDemo')

        self.VISUAL = VISUAL        
        self._cameras = cameras
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
            self.pub_cv =  self.create_publisher(Image, '/roboBreizh_detector/person_feature_detection_image', 10)
            self.pub_compressed_img = self.create_publisher(CompressedImage, "/roboBreizh_detector/person_feature_compressed_image", 10)
            self.pub_compressed_img_age = self.create_publisher(CompressedImage, "/roboBreizh_detector/person_feature_compressed_image_age", 10)
             
    def image_callback(self):

        # entries list of maximum distance for person detection
        distanceMax = 5.0
        
        time_start = time.time()
        # retrieve rgb and depth image from Naoqi camera
        ori_rgb_image, ori_depth_image = self._cameras.get_image(out_format="cv2")
        
        if ori_rgb_image.shape[1] == 640:
            ori_rgb_image_320 = cv2.resize(ori_rgb_image, (320,240))
        else:
            ori_rgb_image_320 = ori_rgb_image
            
        ori_rgb_image_320 = self.colour_detector.apply_brightness_contrast(ori_rgb_image_320)
        
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
                
                if dist <= float(distanceMax):
                    
                    clothes_style = str(detections[0]['class_name'])
                    cropped_clothes = ori_rgb_image_320[int(clothes_start_y):int(clothes_end_y), int(clothes_start_x): int(clothes_end_x)]
                    
                    time_start_clothes = time.time()
                    ok, clothes_color,_, mapping = self.colour_detector.inference(cropped_clothes, 'clothes')     
                                        
                    time_start_face = time.time()
                    # Face Detection
                    output, cropped_face_image, face_start_x, face_start_y, face_end_x, face_end_y = self.face_detector.inference(ori_rgb_image_320)
                    
                    
                    if (cropped_face_image is None):
                        age_caffee = ""
                        gender = ""
                        skin_color = ""
                        self.get_logger().info(
                            bcolors.R+"[RoboBreizh - Vision]    Face not Detected"+bcolors.ENDC)    
                    else:
                        # Face, Age, Gender
                        time_start_gender = time.time()
                        gender  = self.age_gender_detector.inference(cropped_face_image)

                        
                        # Age from Caffee
                        time_start_age = time.time()
                        age_caffee = self.caffe_age_detector.inference(cropped_face_image)

                        time_start_colour = time.time()
                        # Face Skin Colour detection
                        ok, skin_color,_, mapping = self.colour_detector.inference(cropped_face_image, 'skin')

                    if self.VISUAL:
                        image_age_gender = ori_rgb_image_320.copy()
                        # Display clothes
                        cv2.rectangle(ori_rgb_image_320, (int(clothes_start_x), int(clothes_start_y)) , (int(clothes_end_x), int(clothes_end_y)), (255,0,0), 2)
                        # Display face
                        cv2.rectangle(ori_rgb_image_320, (int(face_start_x), int(face_start_y)) , (int(face_end_x), int(face_end_y)), (0,0,255), 2)
                        cv2.rectangle(image_age_gender, (int(face_start_x), int(face_start_y)) , (int(face_end_x), int(face_end_y)), (0,0,255), 2)
                        # display age and gender above face
                        age =  gender + ' ' + age_caffee
                        cv2.addText(image_age_gender, str(age), (int(face_start_x), int(face_start_y-20)), "Arial", (0,0,255), 20)                      
                        
                else:
                    self.get_logger().info(
                        bcolors.R+"[RoboBreizh - Vision]    Clothes/Person Detected but Out of Range"+bcolors.ENDC)                                  
                  
        else:
            self.get_logger().info(
                bcolors.R+"[RoboBreizh - Vision]    Clothes/Person not Detected"+bcolors.ENDC)              
                
        time_end = time.time()

        self.get_logger().info("Total time inference: " + str(time_end-time_start))
        
        if self.VISUAL:
            self.visualiseRVIZ(ori_rgb_image_320, image_age_gender)
 
    def visualiseRVIZ(self, image, image2):
        
        ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pub_cv.publish(ros_image) 
        
        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
        # Publish new image
        self.pub_compressed_img.publish(msg)
        msg2 = CompressedImage()
        msg2.format = "jpeg"
        msg2.data = np.array(cv2.imencode('.jpg', image2)[1]).tostring()
        self.pub_compressed_img_age.publish(msg2)


def main(args=None):
    rclpy.init()

    # print("Starting detection with args: \n model: ", model, "\n resolution: ", res, "\n")
    VISUAL = True
    cameras = NaoqiCameras(ip='127.0.0.1')
    features_detector = PersonFeatureDetection(yolo_model="clothes_320", 
                           face_model="face_detection_yunet_2022mar.onnx", 
                           cafffe_age_model= "age_net.caffemodel",
                           age_gender_model = "AgeGenderTFlite", 
                           glass_model = "shape_predictor_5_face_landmarks.dat", 
                           colour_csv = "new_colorsV3.csv", 
                           cameras=cameras, VISUAL=VISUAL)
    while rclpy.ok():
    # Your code here
        features_detector.image_callback()
    # Clean up when finished
    features_detector.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()
