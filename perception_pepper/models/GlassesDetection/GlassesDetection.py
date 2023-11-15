#!/usr/bin/env python
import time
import cv2
import numpy as np
import dlib
import numpy as np
import cv2
from perception_utils.bcolors import bcolors
import os
from perception_utils.utils import get_pkg_path


class GlassDetection():
    
    def __init__(self, glass_model_name="shape_predictor_5_face_landmarks.dat"):
        
        self.glass_model_name = glass_model_name
        self.glasses_detector = ""
        self.conf_threshold = 0.10
        
    def get_model(self, glass_model_name):
        
        glasses_model_path = os.path.join(get_pkg_path(), ('scripts/models/GlassesDetection/' + glass_model_name))
        glasses_predictor = dlib.shape_predictor(glasses_model_path)
        self.glasses_detector = dlib.get_frontal_face_detector()
     
        return glasses_predictor

    def landmarks_to_np(self, landmarks, dtype="int"):
        
        num = landmarks.num_parts
        
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((num, 2), dtype=dtype)
        
        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, num):
            coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
        # return the list of (x, y)-coordinates
        return coords

    def get_centers(self, img, landmarks):
        
        EYE_LEFT_OUTTER = landmarks[2]
        EYE_LEFT_INNER = landmarks[3]
        EYE_RIGHT_OUTTER = landmarks[0]
        EYE_RIGHT_INNER = landmarks[1]

        x = ((landmarks[0:4]).T)[0]
        y = ((landmarks[0:4]).T)[1]
        A = np.vstack([x, np.ones(len(x))]).T
        k, b = np.linalg.lstsq(A, y, rcond=None)[0]
        
        x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
        x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
        LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
        RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
        
        pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
        # cv2.polylines(img, [pts], False, (255,0,0), 1) 
        # cv2.circle(img, (LEFT_EYE_CENTER[0],LEFT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
        # cv2.circle(img, (RIGHT_EYE_CENTER[0],RIGHT_EYE_CENTER[1]), 3, (0, 0, 255), -1)
        
        return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

    def get_aligned_face(self, img, left, right):
        desired_w = 256
        desired_h = 256
        desired_dist = desired_w * 0.5
        
        eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        dist = np.sqrt(dx*dx + dy*dy)
        scale = desired_dist / dist 
        angle = np.degrees(np.arctan2(dy,dx)) 
        M = cv2.getRotationMatrix2D(eyescenter,angle,scale)

        # update the translation component of the matrix
        tX = desired_w * 0.5
        tY = desired_h * 0.5
        M[0, 2] += (tX - eyescenter[0])
        M[1, 2] += (tY - eyescenter[1])

        aligned_face = cv2.warpAffine(img,M,(desired_w,desired_h))
        
        return aligned_face

    def judge_eyeglass(self, img):
        
        img = cv2.GaussianBlur(img, (11,11), 0) 

        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0 ,1 , ksize=-1) 
        sobel_y = cv2.convertScaleAbs(sobel_y) 
        # cv2.imshow('sobel_y',sobel_y)

        edgeness = sobel_y 
        
        retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        d = len(thresh) * 0.5
        x = np.int32(d * 6/7)
        y = np.int32(d * 3/4)
        w = np.int32(d * 2/7)
        h = np.int32(d * 2/4)

        x_2_1 = np.int32(d * 1/4)
        x_2_2 = np.int32(d * 5/4)
        w_2 = np.int32(d * 1/2)
        y_2 = np.int32(d * 8/7)
        h_2 = np.int32(d * 1/2)
        
        roi_1 = thresh[y:y+h, x:x+w] 
        roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
        roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
        roi_2 = np.hstack([roi_2_1,roi_2_2])
        
        measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])#计算评价值
        measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])#计算评价值
        measure = measure_1*0.3 + measure_2*0.7
        
        print(bcolors.OKGREEN+"     -->  glasses detected confidence: " +
                    str(float(measure)) + bcolors.ENDC)
        
        if measure > self.conf_threshold:
            judge = True
        else:
            judge = False
        
        return judge
       
    def inference(self, rgb_image):
    
        glasses_predictor = self.get_model(self.glass_model_name)
        
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        time_start = time.time()
        rects = self.glasses_detector(gray, 1)
        
        judge = False

        for i, rect in enumerate(rects):
            x_face = rect.left()
            y_face = rect.top()
            w_face = rect.right() - x_face
            h_face = rect.bottom() - y_face
            
            # cv2.rectangle(rgb_image, (x_face,y_face), (x_face+w_face,y_face+h_face), (0,255,0), 2)
            # cv2.putText(rgb_image, "Face #{}".format(i + 1), (x_face - 10, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            landmarks = glasses_predictor(gray, rect)
            landmarks = self.landmarks_to_np(landmarks)
            for (x, y) in landmarks:
                cv2.circle(rgb_image, (x, y), 2, (0, 0, 255), -1)

            LEFT_EYE_CENTER, RIGHT_EYE_CENTER = self.get_centers(rgb_image, landmarks)
            
            aligned_face = self.get_aligned_face(gray, LEFT_EYE_CENTER, RIGHT_EYE_CENTER)            

            judge = self.judge_eyeglass(aligned_face)
            
            # if judge == True:
            #     cv2.putText(rgb_image, "With Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            # else:
            #     cv2.putText(rgb_image, "No Glasses", (x_face + 100, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        time_end = time.time()
        print("Glasses Model Inference time : " + str(time_end - time_start))

        return judge