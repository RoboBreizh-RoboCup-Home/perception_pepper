import cv2
from perception_utils.bcolors import bcolors
import time
from perception_utils.utils import get_pkg_path
import os
import rospy

class SSDInception():
    
    def __init__(self, _conf_threshold):
        
        self.basepath = get_pkg_path()
        self.model_path = "models/ObjectDetection/SSDInception_coco2017/frozen_inference_graph.pb"
        self.config_file = "models/ObjectDetection/SSDInception_coco2017/ssd_inception_v2_coco_2017_11_17.pbtxt"
        self.classes_txt = 'models/ObjectDetection/SSDInception_coco2017/objects.txt'
        self._conf_threshold = _conf_threshold
        self.coco_classes = os.path.join(self.basepath, self.classes_txt)
        self.frozen_model_coco = os.path.join(self.basepath, self.model_path)
        self.config_file_coco = os.path.join(self.basepath, self.config_file)
        
        with open(self.coco_classes, 'rt') as fpt:
            self.coco_classes_list = fpt.read().rstrip('\n').split('\n')
        
        self.cvNet_coco = cv2.dnn.readNetFromTensorflow(
            self.frozen_model_coco, self.config_file_coco)
        
    def inference(self, cv_rgb, object_requested_list):
        t_start_computing = time.time()
        outputs_coco = None
        newDetections_coco = []
        RequestObject = ((object_requested_list)[0] != '')
        
        blob2 = cv2.dnn.blobFromImage(
            cv_rgb, size=(300, 300), swapRB=True, crop=False)
        self.cvNet_coco.setInput(blob2)
        outputs_coco = self.cvNet_coco.forward()
        # ------------------ TIMING  ------------------------------
        t_end_computing = time.time()
        print(bcolors.B+"     --> computeDetections (COCO) delay " +
                str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

        for detection_coco in outputs_coco[0, 0, :, :]:
            score_coco = float(detection_coco[2])
            if score_coco > self._conf_threshold:
                classID_coco = int(detection_coco[1])
                classe_coco = self.coco_classes_list[classID_coco]
                if RequestObject:
                    rospy.loginfo("Detecting Requested objects only")
                    if classe_coco in object_requested_list :
                        newDetections_coco.append(detection_coco)
                    else:
                        rospy.loginfo(
                            bcolors.R+"[RoboBreizh - Vision]        Requested Objects Not in the model class list. "+bcolors.ENDC)   

                else:
                    rospy.loginfo("Detecting all objects")
                    newDetections_coco.append(detection_coco)
                    
        return newDetections_coco