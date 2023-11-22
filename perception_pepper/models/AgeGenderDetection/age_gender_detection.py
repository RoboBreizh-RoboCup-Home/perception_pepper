#!/usr/bin/env python
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2, os
from time import time
from perception_pepper.perception_utils.utils import *
from perception_pepper.perception_utils.utils import get_pkg_path
from perception_pepper.perception_utils.bcolors import bcolors
import time

class AgeGender():
    def __init__(self, age_gender_model = "AgeGenderTFlite") -> None:
        self.string_pred_age = ['20 - 27', '20 - 27','20 - 27','20 - 27','20 - 27','28 - 35','36 - 45','46 - 60','61 - 75']
        self.string_pred_gen = ['F', 'M']
        base_path = get_pkg_path()
        self.age_gender_model_path = os.path.join(base_path, "models/AgeGenderDetection/" + age_gender_model)
        
    def get_model(self, age_gender_model_path):

        # Load TFLite model and allocate tensors. Load Face Cascade
        # face_cascade = cv2.CascadeClassifier(os.path.join(age_gender_model_path, "haarcascade_frontalface_default.xml"))

        # Initialize the TFLite model.
        
        interpreter_age = Interpreter(model_path=os.path.join(age_gender_model_path, "AgeClass_best_06_02-16-02.tflite"))
        interpreter_age.allocate_tensors()

        interpreter_gender = Interpreter(model_path=os.path.join(age_gender_model_path,"GenderClass_06_03-20-08.tflite"))
        interpreter_gender.allocate_tensors()

        # # Get input and output tensors
        input_details_age = interpreter_age.get_input_details()
        output_details_age = interpreter_age.get_output_details()
        input_shape_age = input_details_age[0]['shape']

        input_details_gender = interpreter_gender.get_input_details()
        output_details_gender = interpreter_gender.get_output_details()
        input_shape_gender = input_details_gender[0]['shape']
        
        print(
            bcolors.CYAN+"[RoboBreizh - Vision]        Loading Age/Gender TFlite Detection weights done."+bcolors.ENDC)

        return interpreter_age, input_details_age, output_details_age, input_shape_age, interpreter_gender, input_details_gender, output_details_gender, input_shape_gender

    def inference(self, input_im):
        
        time_start = time.time()
        interpreter_age, input_details_age, output_details_age, input_shape_age, \
            interpreter_gender, input_details_gender, output_details_gender, input_shape_gender = self.get_model(self.age_gender_model_path)        
        
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors=5)        
        
        # if len(faces) == 0:
        #     print("        -->  no face detected, skip face detection")
        #     return None, None

        # our_faces = []
        # for i in faces:
        #     our_faces.append(i)

        # x,y,w,h = faces[0] # We only detect the first face as the frame is the cropped bounding box
        # input_im = frame[y:y+h, x:x+w]

        if input_im is None:
            print("Image is None")
            return None, None
        else:
            input_im = cv2.resize(input_im, (224,224))
            input_im = input_im.astype('float')
            input_im = input_im / 255
            input_im = np.asarray(input_im, dtype='float32')
            if len(input_im.shape) == 2:
                input_im = input_im.reshape((input_im.shape[0], input_im.shape[1], 1))
            input_im = np.expand_dims(input_im, axis = 0)

            # Predict
            input_data = np.array(input_im, dtype=np.float32)
            # interpreter_age.set_tensor(input_details_age[0]['index'], input_data)
            # interpreter_age.invoke()
            interpreter_gender.set_tensor(input_details_gender[0]['index'], input_data)
            interpreter_gender.invoke()

            # output_data_age = interpreter_age.get_tensor(output_details_age[0]['index'])
            output_data_gender = interpreter_gender.get_tensor(output_details_gender[0]['index'])
            # index_pred_age = int(np.argmax(output_data_age))
            index_pred_gender = int(np.argmax(output_data_gender))
            # predic_age = self.string_pred_age[index_pred_age]
            predic_gender = self.string_pred_gen[index_pred_gender]
            
        time_end = time.time()
        
        print("TFlite Age & Gender Model Inference time : " + str(time_end-time_start))

        return predic_gender #predic_age, 


    # def detect_all_faces(self, frame):
    #     font = cv2.FONT_HERSHEY_PLAIN

    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors=5)

    #     for x,y,w,h in faces:
    #         saved_image = frame          
    #         input_im = saved_image[y:y+h, x:x+w]
            
    #         if input_im is None:
    #             print("Image is None")
    #         else:
    #             input_im = cv2.resize(input_im, (224,224))
    #             input_im = input_im.astype('float')
    #             input_im = input_im / 255
    #             input_im = np.asarray(input_im, dtype='float32')
    #             if len(input_im.shape) == 2:
    #                 input_im = input_im.reshape((input_im.shape[0], input_im.shape[1], 1))
    #             input_im = np.expand_dims(input_im, axis = 0)

    #             # Predict
    #             input_data = np.array(input_im, dtype=np.float32)
    #             self.interpreter_age.set_tensor(self.input_details_age[0]['index'], input_data)
    #             self.interpreter_age.invoke()
    #             self.interpreter_gender.set_tensor(self.input_details_gender[0]['index'], input_data)
    #             self.interpreter_gender.invoke()

    #             output_data_age = self.interpreter_age.get_tensor(self.output_details_age[0]['index'])
    #             output_data_gender = self.interpreter_gender.get_tensor(self.output_details_gender[0]['index'])
    #             index_pred_age = int(np.argmax(output_data_age))
    #             index_pred_gender = int(np.argmax(output_data_gender))
    #             predic_age = self.string_pred_age[index_pred_age]
    #             predic_gender = self.string_pred_gen[index_pred_gender]

    #             print("        -->  Age prediction: ", predic_age)
    #             print("        -->  Gender prediction: ", predic_gender)

    #             cv2.putText(frame, predic_age + ', ' + predic_gender, (x,y), font, 1, (255,255,255), 1, cv2.LINE_AA)
    #             cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 1)

    #     return frame