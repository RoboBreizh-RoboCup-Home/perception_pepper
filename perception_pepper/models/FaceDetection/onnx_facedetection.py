#!/usr/bin/env python

import time
import os
import cv2
import numpy as np
from operator import itemgetter

from perception_pepper.perception_utils.utils import get_pkg_path
from perception_pepper.perception_utils.bcolors import bcolors

class FaceDetection():
    
    def __init__(self, face_model_name="face_detection_yunet_2022mar.onnx", score_threshold=0.6, nms_threshold=0.3, input_size=(320, 320)) -> None:
        
        self.face_model_name = face_model_name
        self.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        self.target = cv2.dnn.DNN_TARGET_CPU
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        
    def get_model(self, face_model_name):
        
        face_model_path = os.path.join(get_pkg_path(), ("models/FaceDetection/" + face_model_name))

        return face_model_path
    
    def pre_process(self, face_model_path):
        
        face_detector = cv2.FaceDetectorYN.create(
        model=face_model_path,
        config='',
        input_size= self.input_size,
        score_threshold=self.score_threshold,
        nms_threshold=self.nms_threshold,
        top_k=5000,
        backend_id=self.backend,
        target_id=self.target
        )
        
        return face_detector
    
    def post_process(self, faces, image, display=False):
        
        output = []
        
        if faces is None:
            return [], None, 0, 0, 0, 0

        for idx, face in enumerate(faces):
            coords = face[:-1].astype(np.int32)
            # Crop face
            
            start_y = coords[1]
            end_y = coords[1] + coords[3]
            start_x = coords[0]
            end_x = coords[0] + coords[2]
                        
            # Crop face
            cropped = image[start_y:end_y, start_x:end_x]
            size = cropped.shape[0] * cropped.shape[1]
            
            if not display: # removing landmarks
                coords = coords[:4]
            output.append([coords, size])
            
        # sort results by size of the faces, assume face bigger = closer
        output = sorted(output, key=itemgetter(1))

        # remove the size as we don't care anymore
        for i in range(len(output)):
            output[i] = output[i][0]

            
        return output, cropped, start_x, start_y, end_x, end_y        

    def inference(self, image, verbose=True):
        
        face_model_path = self.get_model(self.face_model_name)
        
        face_detector = self.pre_process(face_model_path)
        
        # Inference
        t_start = time.time()

        face_detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = face_detector.detect(image)
                
        output, cropped, start_x, start_y, end_x, end_y = self.post_process(faces, image)
        
        if verbose:
            t_end = time.time()
            print(bcolors.B+"     -->  face detection delay " +
                    str(round(t_end-t_start, 3)) + bcolors.ENDC)
            print(bcolors.OKGREEN+"     -->  faces detected: " +
                    str(len(output)) + bcolors.ENDC)
        
        return output, cropped, start_x, start_y, end_x, end_y
        
        