#!/usr/bin/python
import rospy

from typing import Dict
from Camera.Naoqi_camera import NaoqiCameras
from models.ObjectDetection.utils import Model
from models.ObjectDetection.MobileNetV2.mobilenet import MobileNetV2
from models.ObjectDetection.InceptionV2.inception import InceptionV2
from robobreizh_msgs.msg import Object, Person
from robobreizh_msgs.srv import object_detection_service


class ObjectDetectionNode():
    def __init__(self, naoqi_camera:NaoqiCameras,object_model:Dict[(str,Model)] ) -> None:
        self._camera = naoqi_camera
        self._is_coco = False
        self._is_oid = False
        self._seat_information = False
        self._bag_information = False
        self._shoes_drink_information = False
        self.models.load_model()
        self.start_services()
        pass

    def start_services(self):
        self.object_detection_service()

    def object_detection_service(self):
        rospy.Service('/robobreizh/perception_pepper/object_detection_service', object_detection_service, self.handle_ServicePerceptionSRV)

    def handle_object_detection_srv(self, srv_msg):
        if srv_msg.maximum_distance == 0:
            srv_msg.maximum_distance = 100


        self.defineStrategy(srv_msg.object_detection_list, srv_msg.publish_person, srv_msg.maximum_distance)

        # get the images 
        images = self._camera.get_image()
        self.rgb_image = images[0]
        self.depth_image = images[1]

        # run detections on every model
        detected_objects = []
        for model in self.models:
            if self._is_oid:
                detected_objects.append(model["oid"].detect(self._camera))
            if self._is_coco:
                detected_objects.append(model["coco"].detect(self._camera))
        print(detected_objects)

    
    def process_image(self, objects_list_requested, publish_person=0, maximum_distance=100):
        pass

    # -----------------------------------------------------------------------------------------------------------------
    def handle_ServicePerceptionHumanSRV(self, person_features_detection_service):
        result = self.process_imageNAOQI(
            person_features_detection_service.entries_list, publishPerson=1)
        return result

    # detectionTask(imageBGR, frame, depth_image, objectsListRequested=objectsRequested, publish_Person=publishPerson, distanceMax=distanceMaximum)
    def detectionTask(self, cv_rgb, frame, depth_image, objectsListRequested, publish_Person=0, distanceMax=100):

        mappingImages = []

        self.objects_detected_msg: Object() = []
        self.person_detected_msg: Person() = []

        width = cv_rgb.shape[1]
        height = cv_rgb.shape[0]

        self.defineStrategy(objectsListRequested, publish_Person, distanceMax)

        outputs_oid, outputs_coco = self.computeDetections(cv_rgb, bCOCO, bOID)

        # get objects according to request
        newDetections_oid = self.matchWithRequestOID(
            outputs_oid, objectsListRequested, bAll, bseat_information, bbag_information, _shoes_drink_information, bOID)
        newDetections_coco = self.matchWithRequestCOCO(
            outputs_coco, objectsListRequested, bAll, bseat_information, bbag_information, _shoes_drink_information, bCOCO)

        # Coco vs OID --> cleaning
        newDetections2_coco = self.adjustCoco(
            bOID, bCOCO, newDetections_coco, newDetections_oid, width, height)
        # colors, distance ....
        self.computeModelsOnAllCOCO(newDetections2_coco, width, height,
                                    frame, depth_image, mappingImages, distanceMax, publish_Person)

        # OID vs COCO --> cleaning
        newDetections2_oid = self.adjustOID(
            bOID, bCOCO, newDetections2_coco, newDetections_oid, width, height)
        # colors, distance ....
        self.computeModelsOnAllOID(newDetections2_oid, width, height,
                                   frame, depth_image, mappingImages, distanceMax, publish_Person)

        # seat / shoes / drink / bag
        self.manageSeatTaken(bseat_information, bOID, bCOCO, newDetections2_oid,
                             newDetections2_coco, width, height, depth_image)
        mess = self.manageShoesDrink(
            _shoes_drink_information, newDetections2_oid, newDetections2_coco, width, height)
        self.manageBag(bbag_information, bOID, bCOCO, newDetections2_oid,
                       newDetections2_coco, width, height, depth_image)

        # ###### publish Image with detection + distance + color ################
        if DISPLAY == 1:
            t_start_publish = time.time()
            self.publishDisplayImage(frame, depth_image)
            if DETECT_COLOR == 1:
                self.publishDisplayColor(mappingImages)
            t_end_publish = time.time()
            if DISPLAY_ALL_DELAY == 1:
                print(bcolors.B+"     --> publish images delay " +
                      str(round(t_end_publish - t_start_publish, 3))+bcolors.ENDC)

       # ------------------ TIMING  ------------------------------
        t_end_computing = time.time()
        if DISPLAY_DELAY == 1:
            print(bcolors.B+"     --> TOTAL detection delay " +
                  str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision] detection done "+bcolors.ENDC)

        # ------------------ RViz Marker ------------------------------
        if self.objectsDetectedMANAGER.objects_list and DISPLAY == 1:
            # print(self.objectsDetectedMANAGER.objects_list)
            self.show_RViz_marker_arr(self.objectsDetectedMANAGER.objects_list, DURATION=50)

        ###### Service ################

        if (publish_Person == 1):
            return self.personsDetectedMANAGER
        elif self._shoes_drink_information == 1:
            return mess
        else:
            return self.objectsDetectedMANAGER

    def defineStrategy(self, objectsListRequested, publish_Person, distanceMax):
        # --------- SEAT -----
        for object in objectsListRequested:
            if "SEAT_INFORMATION" == object.data:
                self._seat_information = True
                self._is_oid = True
                print("     ask for SEAT_INFORMATION detections ....")
            elif "BAG_INFORMATION" == object.data:
                self._bag_information = True
                self._is_coco = True
                self._is_oid = True
                print("     ask for BAG_INFORMATION detections ....")
            # --------- SHOES and DRINK -----
            elif "SHOES_DRINK_INFORMATION" == object.data:
                self._shoes_drink_information= True
                self._is_coco = True
                self._is_oid = True
                print("     ask for SHOES_DRINK_INFORMATION detections ....")
            # --------- ALL -----
            elif "ALL" == object.data:
                self._is_coco = True
                self._is_oid = True
                print("     ask for ALL detections....")
            else:
                self._is_coco = True
                self._is_oid = True
                print("     ask classical detections with resquested objects ...")

        # --------- PERSON -----
        if publish_Person == 1:
            if distanceMax == 100:
                print("     ask for Persons detections ....")
            else:
                print("     ask for Persons detections with distance filter ....")
            self._is_coco = True

    def computeDetections(self, cv_rgb):
        t_start_computing = time.time()
        outputs = None
        outputs_coco = None

        # detect object
        if self._is_oid:
            blob = cv2.dnn.blobFromImage(cv_rgb, size=(300, 300), swapRB=False, crop=False)
            self.cvNet.setInput(blob)
            outputs = self.cvNet.forward()

            # ------------------ TIMING  ------------------------------
            t_end_computing = time.time()
            if DISPLAY_DELAY == 1:
                print(bcolors.B+"     --> computeDetections (OID) delay " +
                      str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)
                t_start_computing = time.time()

        if bCOCO == 1:
            blob2 = cv2.dnn.blobFromImage(
                cv_rgb, size=(300, 300), swapRB=True, crop=False)
            self.cvNet_coco.setInput(blob2)
            outputs_coco = self.cvNet_coco.forward()
            # ------------------ TIMING  ------------------------------
            t_end_computing = time.time()
            if DISPLAY_DELAY == 1:
                print(bcolors.B+"     --> computeDetections (COCO) delay " +
                      str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

        return outputs, outputs_coco


# main function
if "__name__" == "__main__":
    rospy.init_node("robobreizh_object_detection_node")
    cameras = NaoqiCameras()
    object_detection_model = {"oid":MobileNetV2(), "coco":InceptionV2()}
    ObjectDetectionNode(cameras, object_model=object_detection_model)
    rospy.spin()
