
#!/usr/bin/env python
import os
import cv2
from perception_utils.utils import get_pkg_path
import time
from perception_utils.bcolors import bcolors
import rospy

class AgePrediction():
    
    def __init__(self, age_model_name = "age_net.caffemodel"):
        
        self.age_model_name = age_model_name
        self.ageList = ['(15-20)', '(15-20)', '(15-20)', '(15-20)',
                    '(25-32)', '(38-43)', '(48-53)', '(48-53)']
    
    def get_model(self, age_model_name):
        # Age Detection model
        age_detector = cv2.dnn.readNetFromCaffe(
                os.path.join(
                    get_pkg_path(), "scripts/models/AgeGenderDetection/age/deploy_age.prototxt"),
                os.path.join(get_pkg_path(), ("scripts/models/AgeGenderDetection/age/" + age_model_name))
            ) 
        
        return age_detector    

    def inference(self, cropped):
        """
        :return: ageRange (string)
        """
        
        age_detector = self.get_model(self.age_model_name)
        
        age_prediction = ''
        ageRange = ''

        start_time = time.time()
        if cropped.size != 0:
            blob_age = cv2.dnn.blobFromImage(cropped, 1, (227, 227))
            age_detector.setInput(blob_age)
            age_prediction = age_detector.forward()

            ageRange = self.ageList[age_prediction[0].argmax()]
            end_time = time.time()
        else:
            print(
                bcolors.WARNING+'     ObjectDetector agePrediction cropped image  EMPTY'+bcolors.ENDC)
        rospy.loginfo("Caffee Age Model Inference time : " + str(end_time-start_time))
        
        return ageRange