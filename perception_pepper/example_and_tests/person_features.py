#!/usr/bin/env python
# import roslib
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

# import models
from perception_pepper.models.ObjectDetection.YOLOV8 import YOLOV8
from perception_pepper.models.FaceDetection.onnx_facedetection import FaceDetection
from perception_pepper.models.AgeGenderDetection.age_gender_detection import AgeGender
from perception_pepper.models.AgeGenderDetection.CaffeAgeDetection import AgePrediction
from perception_pepper.models.GlassesDetection.GlassesDetection import GlassDetection
from perception_pepper.models.ColourDetection.ColourDetection import ColourDetection

# import utils
from perception_pepper.perception_utils.bcolors import bcolors
import perception_pepper.perception_utils.distances_utils as distances_utils
import time

class PersonFeatureDetection(Node):
        
    def __init__(self , yolo_model, face_model, cafffe_age_model, age_gender_model, glass_model, colour_csv, VISUAL) -> None:
        super().__init__('FeaturesDemo')

        self.VISUAL = VISUAL        
        self.conf_threshold = 0.5
        self.nms_threshold = 0.5
    
        self.yolo_clothes_detector = YOLOV8(model_name=yolo_model,  _conf_threshold=self.conf_threshold, _iou_threshold=self.nms_threshold)
        self.face_detector = FaceDetection(face_model_name = face_model)
        self.age_gender_detector = AgeGender(age_gender_model = age_gender_model)
        self.caffe_age_detector = AgePrediction(age_model_name = cafffe_age_model)
        self.glass_detector = GlassDetection(glass_model_name=glass_model) 
        self.colour_detector = ColourDetection(colour_csv_file_name = colour_csv, color_type="hue")

        self.bridge = CvBridge()

        if self.VISUAL:
            self.pub_cv =  self.create_publisher(Image, '/roboBreizh_detector/person_feature_detection_image', 10)
        
        self.subscriber = self.create_subscription(Image, '/image_raw', self.image_callback, 10)

    def draw_bbox(self, image, bbox, label, color=(255, 0, 0), thickness=2):
        """Draws single bounding box on the image"""
        x1, y1, x2, y2 = bbox
        image = cv2.rectangle(image, (int(x1), int(y1)) , (int(x2), int(y2)), color, thickness)
        # display cloth name in black text
        text_color = (255, 255, 255)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Prints the text.
        image = cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        image = cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        return image
        
    def image_callback(self, data):
        
        time_start = time.time()
        # retrieve rgb and depth image from Naoqi camera
        ori_rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        
        if ori_rgb_image.shape[1] == 640:
            ori_rgb_image_320 = cv2.resize(ori_rgb_image, (320,240))
        else:
            ori_rgb_image_320 = ori_rgb_image
            
        # ori_rgb_image_320 = self.colour_detector.apply_brightness_contrast(ori_rgb_image_320)
        
        # Clothes Detection
        detections = self.yolo_clothes_detector.inference(ori_rgb_image_320)

        if (len(detections) > 0):
            clothes_start_x = round(detections[0]['box'][0])
            clothes_end_x = round((detections[0]['box'][0] + detections[0]['box'][2]))
            clothes_start_y = round(detections[0]['box'][1])
            clothes_end_y = round((detections[0]['box'][1] + detections[0]['box'][3]))
            
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
                # Display clothes in red rectangle
                image_clothes = self.draw_bbox(ori_rgb_image_320, [clothes_start_x, clothes_start_y, clothes_end_x, clothes_end_y], clothes_color+' '+clothes_style, color=(0, 0, 255), thickness=2)

                # Display face
                age_label =  gender + ' ' + age_caffee
                image_age_gender = self.draw_bbox(image_clothes, [face_start_x, face_start_y, face_end_x, face_end_y], age_label, color=(255, 0, 0), thickness=2)                                           
                  
        else:
            self.get_logger().info(
                bcolors.R+"[RoboBreizh - Vision]    Clothes/Person not Detected"+bcolors.ENDC)    
            image_age_gender = ori_rgb_image_320.copy()
                
        time_end = time.time()

        self.get_logger().info("Total time inference: " + str(time_end-time_start))
        
        if self.VISUAL:
            self.visualiseRVIZ(image_age_gender)
 
    def visualiseRVIZ(self, image):
        
        ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.pub_cv.publish(ros_image)


def main(args=None):
    rclpy.init()

    # print("Starting detection with args: \n model: ", model, "\n resolution: ", res, "\n")
    VISUAL = True

    features_detector_node = PersonFeatureDetection(yolo_model="clothes_320", 
                           face_model="face_detection_yunet_2022mar.onnx", 
                           cafffe_age_model= "age_net.caffemodel",
                           age_gender_model = "AgeGenderTFlite", 
                           glass_model = "shape_predictor_5_face_landmarks.dat", 
                           colour_csv = "new_colorsV3.csv",
                           VISUAL=VISUAL)
    try:
        rclpy.spin(features_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        features_detector_node.destroy_node()
        rclpy.shutdown()
    
if __name__ == "__main__":
    main()
