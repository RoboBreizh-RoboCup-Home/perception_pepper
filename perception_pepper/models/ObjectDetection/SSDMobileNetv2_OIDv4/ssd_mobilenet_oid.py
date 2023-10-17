import cv2
from perception_utils.bcolors import bcolors
import time
from perception_utils.utils import get_pkg_path
import os
import rospy

class SSDMobileNetV2():
    
    def __init__(self, _conf_threshold):
        
        self.basepath = get_pkg_path()
        self.model_path = "scripts/models/ObjectDetection/SSDMobileNetv2_OIDv4/frozen_inference_graph.pb"
        self.config_file = "scripts/models/ObjectDetection/SSDMobileNetv2_OIDv4/graph.pbtxt"
        self.classes_txt = 'scripts/models/ObjectDetection/SSDMobileNetv2_OIDv4/objects.names.en'
        self._conf_threshold = _conf_threshold
        self.oid_classes = os.path.join(self.basepath, self.classes_txt)
        self.frozen_model_oid = os.path.join(self.basepath, self.model_path)
        self.config_file_oid = os.path.join(self.basepath, self.config_file)
        
        with open(self.oid_classes, 'rt') as fpt:
            self.oid_classes_list = fpt.read().rstrip('\n').split('\n')
        
        self.cvNet_oid = cv2.dnn.readNetFromTensorflow(
            self.frozen_model_oid, self.config_file_oid)
        
    def inference(self, cv_rgb, object_requested_list):
        t_start_computing = time.time()
        outputs_oid = None
        newDetections_oid = []
        RequestObject = ((object_requested_list)[0] != '')
        
        blob2 = cv2.dnn.blobFromImage(
            cv_rgb, size=(300, 300), swapRB=True, crop=False)
        self.cvNet_oid.setInput(blob2)
        outputs_oid = self.cvNet_oid.forward()
        # ------------------ TIMING  ------------------------------
        t_end_computing = time.time()
        print(bcolors.B+"     --> computeDetections (oid) delay " +
                str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

        for detection_oid in outputs_oid[0, 0, :, :]:
            score_oid = float(detection_oid[2])
            if score_oid > self._conf_threshold:
                classID_oid = int(detection_oid[1])
                classe_oid = self.oid_classes_list[classID_oid-1]
                if RequestObject:
                    rospy.loginfo("Detecting Requested objects only")
                    if classe_oid in object_requested_list :
                        newDetections_oid.append(detection_oid)
                    else:
                        rospy.loginfo(
                            bcolors.R+"[RoboBreizh - Vision]        Requested Objects Not in the model class list. "+bcolors.ENDC)   

                else:
                    rospy.loginfo("Detecting all objects")
                    newDetections_oid.append(detection_oid)
                    
        return newDetections_oid