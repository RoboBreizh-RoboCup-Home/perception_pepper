#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import rospy
import numpy as np
import message_filters

# opencv
import cv2
from cv_bridge import CvBridge

# camera
# import Camera.Naoqi_camera as nc
# from Camera.naoqi_camera_types import CameraID, CameraResolution2D as res2D, CameraResolution3D as res3D, ColorSpace2D as cs2D, ColorSpace3D as cs3D

# ros message
from std_msgs.msg import String, Header, Int64
from sensor_msgs.msg import Image

# ros service
from robobreizh_msgs.srv import litter_detection
    

class LitterDetection():
    """
    A class used to detect litter on the floor.
    """
    
    def __init__(self):
        self.bridge = CvBridge() # For handle cv2 image
        
        # PUBLISHERS
        # The following publisher and subcribers are not used for service node
        self.rgb_image_pub = rospy.Publisher(
            "/detected_rgb_image", Image, queue_size=10)
        self.bin_image_pub = rospy.Publisher(
            "/binary_image", Image, queue_size=10)
        
        # self.init_service()


    def init_service(self):
        """
        Service node for litter detection
        """
        
        rospy.Service('/robobreizh/perception_pepper/litter_detection', litter_detection,
                          self.handle_service)
        rospy.loginfo("Starting Litter Detection Service: Waiting for Request...")
        rospy.spin()


    def handle_service(self, req):
        """
        Apply contour detection and calculating contour area to identify litter from input rgb image.
        
        :param (ros.srv - litter_detection) req: The request is the threshold for binary image
        :return (ros.msg - bool) is_litter: True if there are contours in the frame, False otherwise
        """
        
        is_litter = False
        time_begin = rospy.Time.now()
        rospy.loginfo("HANDLE REQUEST")
        
        image_sub = rospy.wait_for_message('/naoqi_driver/camera/front/image_raw', Image)
        rgb_img = self.bridge.imgmsg_to_cv2(image_sub, "bgr8")
        
        # Convert to binary image
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray_img, 127, 255, 0)
        # Find contours to extract litter from the floor
        contours, _ = cv2.findContours(binary_img,
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Identify litter based on contour area
        for cnt in contours:
            # Calculate the area of the contour
            cnt_area = cv2.contourArea(cnt)
            if 5 <= cnt_area <= 10:
                rospy.loginfo("THERE IS LITTER ON THE FLOOR")
                is_litter = True
                
        
        time_end = rospy.Time.now()
        duration = time_end - time_begin
        rospy.loginfo(f"PROCESSING TIME: {duration.to_sec()} secs")
        
        return is_litter
        

    def continuous_node(self):
        """
        Publishing node for testing litter detection 
        """

        while not rospy.is_shutdown():
            time_begin = rospy.Time.now()
            rospy.loginfo("START PROCESSING")
            image_sub = rospy.wait_for_message('/naoqi_driver/camera/front/image_raw', Image)
            rgb_img = self.bridge.imgmsg_to_cv2(image_sub, "bgr8")
            img_h, img_w = rgb_img.shape[:2]
            crop_floor_img = rgb_img[int(1/3*img_h):, int(1/5*img_w): int(4/5*img_w), :]
            
            # Convert to binary image
            gray_img = cv2.cvtColor(crop_floor_img, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(gray_img, 127, 255, 0)
            # Find contours to extract litter from the floor
            contours, _ = cv2.findContours(binary_img, 
                                        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Identify litter based on contour area
            output_img = crop_floor_img.copy()
            boxes = []
            scores = []
            for cnt in contours:
                # Calculate the area of the contour
                cnt_area = cv2.contourArea(cnt)
                if 5 <= cnt_area <= 10:
                    x, y, w, h = cv2.boundingRect(cnt)
                    boxes.append([x, y, w, h])
                    scores.append(cnt_area)
                    
            # indexes = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=6, nms_threshold=0.3)
            for index in len(boxes):  
                x, y, w, h = boxes[index]
                cnt_area = scores[index]
                output_img = cv2.rectangle(output_img, (x, y), (x + w, y + h), (255,0,0), 2)
                output_img = cv2.putText(output_img, str(cnt_area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                                                0.5, (0,255,0), 1, cv2.LINE_AA)
            
            rospy.loginfo("FINISH DETECT LITTER")
            output_msg = self.bridge.cv2_to_imgmsg(output_img, "bgr8")
            bin_msg = self.bridge.cv2_to_imgmsg(binary_img, "mono8")
            self.rgb_image_pub.publish(output_msg)
            self.bin_image_pub.publish(bin_msg)
            
            time_end = rospy.Time.now()
            duration = time_end - time_begin
            rospy.loginfo(f"PROCESSING TIME: {duration.to_sec()} secs")
        
        


if __name__ == "__main__":
    
    rospy.init_node('litter_detection_node', anonymous=False)

    # VISUAL = rospy.get_param('~visualize')
    # qi_ip = rospy.get_param('~qi_ip')
    
    # depth_camera_res = res3D.R320x240
    # rgb_camera_res = res2D.R320x240

    # cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
    
    # PersonFeatureDetection(yolo_model="clothes_320", 
    #                        face_model="face_detection_yunet_2022mar.onnx", 
    #                        cafffe_age_model= "age_net.caffemodel",
    #                        age_gender_model = "AgeGenderTFlite", 
    #                        glass_model = "shape_predictor_5_face_landmarks.dat", 
    #                        colour_csv = "new_colorsV2.csv", 
    #                        cameras=cameras, VISUAL=VISUAL)
    
    detection_node = LitterDetection()

    # Here you can switch between two mode:
    # 1. Continuous detection by ROS subscriber/callback (asynchronous)
    # 2. Synchronous detection via ROS Service (Server/Client-like)
    try:
        detection_node.continuous_node()
    except rospy.ROSInterruptException:
        pass
    
    # detection_node.service_node()
