#! /usr/bin/env python
# ----------------------------------------------------------------------------
# Authors  : Cedric BUCHE (buche@enib.fr)
# Created Date: 2022
# ---------------------------------------------------------------------------

import numpy as np
import sys
import time
import sys
import os

# ROS
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32, Pose, PoseArray

# OPENCV
import cv2
from cv_bridge import CvBridge

# NAOQI
import qi

# ROBOBREIZH
from perception_utils.bcolors import bcolors
from perception_utils.utils import *
from perception_utils.transform_utils import *
from perception_utils.display_utils import *
from perception_utils.distances_utils import *
from perception_utils.object_detection_utils import *
from perception_utils.colors_utils import *
from PoseDetection.MoveNet_MultiPose.movenet_multipose import MoveNetMultiPose
from perception_utils.roboBreizh_Utils_pose import *
from perception_utils.multipose_service import Pose_Detector
from AgeDetection.age_gender_detection import AgeGender
from robobreizh_msgs.msg import *
from robobreizh_msgs.srv import *

# RViz Visualization using Marker
from visualization_msgs.msg import Marker, MarkerArray

from ObjectDetection import YOLOV8, MobileNetV2, InceptionV2, YOLOV5

from typing import List

#########################################################################################
# variables
#########################################################################################
global DESCRIBE_PERSON
global DETECT_AGE
global DETECT_COLOR
global DETECT_DISTANCE

global POSTURE_PERSON
global WAVE_HAND
global POINTING_HAND

global DISPLAY_DELAY
global DISPLAY_ALL_DELAY
global DISPLAY

#########################################################################################


#########################################################################################
class RobobreizhCV_Detector():
    def __init__(self, session=None):

        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision] loading ..."+bcolors.ENDC)

        # NaoQI
        self.session = session
        self.initCamerasNaoQi()

        # ROS <-> CV2
        self.bridge = CvBridge()

        # Models
        self.loadModels()

        # services
        self.defineServices()

        # display on PC
        self.initDisplayPublisher()

        # spin
        rospy.on_shutdown(self.cleanup)		# What we do during shutdown

        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision] Waiting for service requests ..."+bcolors.ENDC)
        rospy.spin()

# -----------------------------------------------------
    def loadModels(self):

        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision] Loading models ... "+bcolors.ENDC)

        # human's posture
        self.initTransformation()
        self.init_posture_model()

        # object
        self.initObjectDetector_OID()
        self.initObjectDetector_coco_2017_kitchen()

        # age estimation
        self.initAgeDetector()

        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision] Loading models done "+bcolors.ENDC)


# -----------------------------------------------------

    def defineServices(self):

        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision] Starting services ... "+bcolors.ENDC)

        # human's posture
        # self.initPostureDetectionService()
        # self.initWavingDetectionService()
        # self.initPointingDetectionService()

        # object
        self.initObjectsDetectionService()

        # person
        # self.initSimplePersonDetectionService()
        self.initPersonDescriptionService()
        self.initPersonDescriptionWithDistanceService()

        # seat information
        self.initSeatDetectionService()

        # bag information
        self.initBagDetectionService()

        # shoes and drink information
        self.initShoesDrinkDetectionService()

        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision] Starting services done "+bcolors.ENDC)

    ######################################################################
    # NAOQI CAM
    ######################################################################

    def initCamerasNaoQi(self):
        self.video_service = session.service("ALVideoDevice")
        fps = 2
        resolutionD = 1  	# Image of 320*240px
        colorSpaceD = 17  	# mono16
        # resolution = 1  	# Image of 320*240px
        resolution = 2  	# Image of 320*240px
        colorSpace = 11  	# RGB
        self.videosClient = self.video_service.subscribeCameras(
            "cameras_pepper", [0, 2], [resolution, resolutionD], [colorSpace, colorSpaceD], fps)

    ######################################################################
    # POSTURE
    ######################################################################

      # -----------------------------------------------------------------------------------------------------------------

    # def initPostureDetectionService(self):
        # if POSTURE_PERSON == 1:
        #     self.age_gender_detect = AgeGender()
        #     s2 = rospy.Service('/robobreizh/perception_pepper/person_features_detection_posture',
        #                        person_features_detection_posture, self.handle_ServicePerceptionHumanSRVWithDistancePosture)
        #     rospy.loginfo(
        #         bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] Launch Age + Posture service..."+bcolors.ENDC)
    # -----------------------------------------------------------------------------------------------------------------

    # def initWavingDetectionService(self):
    #     if WAVE_HAND == 1:
    #         s2 = rospy.Service('/robobreizh/perception_pepper/wave_hand_detection',
    #                            wave_hand_detection, self.handle_ServiceWaveHandDetection)
    #         rospy.loginfo(
    #             bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] Launch Wave Hand Detection service..."+bcolors.ENDC)
    # -----------------------------------------------------------------------------------------------------------------

    # def initPointingDetectionService(self):
    #     if POINTING_HAND == 1:
    #         s2 = rospy.Service('/robobreizh/perception_pepper/pointing_hand_detection',
    #                            pointing_hand_detection, self.handle_ServicePointingHandDetection)
    #         rospy.loginfo(
    #             bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] Launch Pointing Hand Detection service..."+bcolors.ENDC)

    # -----------------------------------------------------------------------------------------------------------------

    # def init_posture_model(self):

    #     if WAVE_HAND == 1 or POSTURE_PERSON == 1 or POINTING_HAND == 1:
    #         rospy.loginfo(bcolors.ON_PURPLE +
    #                       "[RoboBreizh - Pose/Vision]     -> Loading Posture models.."+bcolors.ENDC)
    #         # self.pose_detection = Pose_Detector(self.session, self.transform_camera_map)
    #         # self.pose_detector = MoveNetMultiPose("movenet_multipose", None)
    #         rospy.loginfo(
    #             bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision]     -> Loading Posture models done"+bcolors.ENDC)

    ######################################################################
    # Objects
    ######################################################################
    def initObjectDetector_OID(self):
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]     -> Loading Detector weights - OID v4..."+bcolors.ENDC)
        pkg_path = get_pkg_path()
        config_file = os.path.join(
            pkg_path, 'scripts/models/ssd_mobilenet_v2_oid_v4/graph.pbtxt')
        frozen_model = os.path.join(
            pkg_path, 'scripts/models/ssd_mobilenet_v2_oid_v4/frozen_inference_graph.pb')
        file_name = os.path.join(
            pkg_path, 'scripts/models/ssd_mobilenet_v2_oid_v4/objects.names.en')
        with open(file_name, 'rt') as fpt:
            self.classes = fpt.read().rstrip('\n').split('\n')
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.cvNet = cv2.dnn.readNetFromTensorflow(frozen_model, config_file)
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]        Loading Detector weights done - OID v4."+bcolors.ENDC)

    # -----------------------------------------------------------------------------------------------------------------
    def initObjectDetector_coco_2017_kitchen(self):
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]     -> Loading Detector weights - COCO 2017..."+bcolors.ENDC)
        base_path = get_pkg_path()
        config_file_coco = os.path.join(
            base_path, 'scripts/models/ssd_inception_v2_coco_2017/ssd_inception_v2_coco_2017_11_17.pbtxt')
        frozen_model_coco = os.path.join(
            base_path, 'scripts/models/ssd_inception_v2_coco_2017/frozen_inference_graph.pb')
        file_name_coco = os.path.join(
            base_path, 'scripts/models/ssd_inception_v2_coco_2017/objects.txt')
        with open(file_name_coco, 'rt') as fpt:
            self.classes_coco = fpt.read().rstrip('\n').split('\n')
        self.colors_coco = np.random.uniform(
            0, 255, size=(len(self.classes_coco), 3))
        self.cvNet_coco = cv2.dnn.readNetFromTensorflow(
            frozen_model_coco, config_file_coco)
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]        Loading Detector weights done - COCO 2017."+bcolors.ENDC)

    # -----------------------------------------------------------------------------------------------------------------
    def initObjectsDetectionService(self):
        rospy.Service('/robobreizh/perception_pepper/object_detection_service',
                      object_detection_service, self.handle_ServicePerceptionSRV)
    # -----------------------------------------------------------------------------------------------------------------

    def initShoesDrinkDetectionService(self):
        rospy.Service('/robobreizh/perception_pepper/shoes_and_drink_detection_service',
                      shoes_and_drink_detection_service, self.handle_ServicePerceptionShoesDrinkSRV)

    # -----------------------------------------------------------------------------------------------------------------
    def initSeatDetectionService(self):
        rospy.Service('/robobreizh/perception_pepper/seat_detection_service',
                      object_detection_service, self.handle_ServicePerceptionSeatSRV)

    # -----------------------------------------------------------------------------------------------------------------
    def initBagDetectionService(self):
        rospy.Service('/robobreizh/perception_pepper/bag_detection_service',
                      object_detection_service, self.handle_ServicePerceptionBagSRV)

    ######################################################################
    # PERSON
    ######################################################################

    def createPerson(self, classe):
        person = Person()
        person.name = ""
        person.clothes_color = ""
        person.age = ""
        person.gender = ""
        person.skin_color = ""
        person.height = 0.0
        person.distance = 0.0
        person.coord.x = 0
        person.coord.y = 0
        person.coord.z = 0

        if classe == "Man":
            person.gender = "M"
        elif classe == "Woman":
            person.gender = "F"

        return person
    # -----------------------------------------------------------------------------------------------------------------

    def initPersonDescriptionService(self):
        if DESCRIBE_PERSON == 1:
            rospy.Service('/robobreizh/perception_pepper/person_features_detection_service',
                          person_features_detection_service, self.handle_ServicePerceptionHumanSRV)
    # -----------------------------------------------------------------------------------------------------------------

    def initPersonDescriptionWithDistanceService(self):
        if DESCRIBE_PERSON == 1:
            rospy.Service('/robobreizh/perception_pepper/person_features_detection_distance_service',
                          person_features_detection_distance_service, self.handle_ServicePerceptionHumanSRVWithDistance)

    ######################################################################
    # Display
    ######################################################################
    def initDisplayPublisher(self):
        if DISPLAY == 1:
            self.pub_cv = rospy.Publisher(
                '/roboBreizh_detector/perception_object_2d', Image, queue_size=1)
            self.pub_cv_depth = rospy.Publisher(
                '/roboBreizh_detector/perception_D', Image, queue_size=1)
            self.pub_kmeans = rospy.Publisher(
                '/roboBreizh_detector/perception_color', Image, queue_size=1)
            # self.pub_visu = rospy.Publisher(
            #     '/roboBreizh_detector/perception_multipose', Image, queue_size=1)

            # Publisher for Markers
            self.marker_arr_pub = rospy.Publisher(
                '/roboBreizh_detector/visualization_marker_array', MarkerArray, queue_size=1)

    # -----------------------------------------------------------------------------------------------------------------
    def publishDisplayImage(self, cv_rgb, cv_depth):
        if cv_depth.any():  # != None :
            perception_message_depth = self.bridge.cv2_to_imgmsg(
                cv_depth, "32FC1")
            self.pub_cv_depth.publish(perception_message_depth)

        if cv_rgb.any():  # != None :
            perception_message = self.bridge.cv2_to_imgmsg(cv_rgb, "bgr8")
            self.pub_cv.publish(perception_message)

    # -----------------------------------------------------------------------------------------------------------------
    def publishDisplayColor(self, mappingImages):
        max_height = 0
        for i in range(len(mappingImages)):
            mapping = mappingImages[i]
            if (mapping.shape[0] > max_height):
                max_height = mapping.shape[0]

        max_height += 10

        for i in range(len(mappingImages)):
            mapping = mappingImages[i]
            top2 = int((max_height - mapping.shape[0]) / 2)
            bottom2 = max_height - mapping.shape[0] - top2
            if (top2 < 0):
                top2 = 0
            if (bottom2 < 0):
                bottom2 = 0
            mapping = cv2.copyMakeBorder(
                mapping, top2, bottom2, 5, 5, cv2.BORDER_CONSTANT)
            if (i == 0):
                mapping_concat = mapping
            else:
                mapping_concat = np.concatenate(
                    (mapping_concat, mapping), axis=1)

        if len(mappingImages) != 0:
            kmeans_message = self.bridge.cv2_to_imgmsg(
                mapping_concat, "bgr8")
        if len(mappingImages) != 0:
            self.pub_kmeans.publish(kmeans_message)

    ######################################################################
    # AGE
    ######################################################################

    def initAgeDetector(self):
        if DETECT_AGE == 1:
            rospy.loginfo(
                bcolors.CYAN+"[RoboBreizh - Vision]     -> Loading Age weights..."+bcolors.ENDC)
            pkg_path = get_pkg_path()
            self.age_model = cv2.dnn.readNetFromCaffe(
                os.path.join(
                    pkg_path, "scripts/models/age/deploy_age.prototxt"),
                os.path.join(pkg_path, "scripts/models/age/age_net.caffemodel")
            )
            rospy.loginfo(
                bcolors.CYAN+"[RoboBreizh - Vision]        Loading Age weights done."+bcolors.ENDC)

    # -----------------------------------------------------------------------------------------------------------------
    def agePrediction(self, cropped):
        """
        :return: ageRange (string)
        """
        age_prediction = ''
        ageRange = ''

        if DETECT_AGE == 1:
            t40 = time.time()
            if cropped.size != 0:
                blob_age = cv2.dnn.blobFromImage(cropped, 1, (227, 227))
                self.age_model.setInput(blob_age)
                age_prediction = self.age_model.forward()
                ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
                ageRange = ageList[age_prediction[0].argmax()]
                t50 = time.time()
                if DISPLAY_ALL_DELAY == 1:
                    print(bcolors.B+"     -->  age detection delay " +
                          str(round(t50-t40, 3)) + bcolors.ENDC)
            else:
                print(
                    bcolors.WARNING+'     ObjectDetector agePrediction cropped image  EMPTY'+bcolors.ENDC)
        return ageRange

    # -----------------------------------------------------------------------------------------------------------------
    def manageAgeOID(self, cropped, classe):
        """
        :return: text_age (string)
        """
        text_age = ''
        if DETECT_AGE == 1:
            if classe == "Human face":
                if cropped.size != 0:
                    age = self.agePrediction(cropped)
                    text_age = "--- " + str(age) + ' years'
        return text_age

    ######################################################################
    # COLOR
    ######################################################################

    def colorDetection_withImage(self, cropped, classe):
        """
        :return: ok (boolean), color_res (string) , mapping (image)
        """
        color_res = ''
        t400 = time.time()
        color_resize_factor = 10
        data = new_csv_reader(newest_csv_path)
        row_count = sum(1 for line in data)
        colors = np.zeros(shape=(row_count, 3))
        for i in range(len(data)):
            ref_color = data[i]
            colors[i] = list(map(int, ref_color[0].split('-')[:3]))
        if DISPLAY == 1:
            color_resize_factor = 15
        if len(cropped) != 0:
            ok, color_res, center_sorted, mapping = detect_colors(
                cropped, num_clusters=5, num_iters=50, resize_factor=color_resize_factor, crop_factor=50, colors_in_order=colors, csv=data, type="rgb", name=str(classe))
        else:
            return 0, '', []
        if ok:
            t500 = time.time()
            if DISPLAY_ALL_DELAY == 1:
                print(bcolors.B+"     -->  color detection delay " +
                      str(round(t500-t400, 3)) + ' ('+(str(classe))+')'+bcolors.ENDC)
        return ok, color_res, mapping

    # -----------------------------------------------------------------------------------------------------------------
    def colorDetection(self, cropped, classe):
        """
        :return: ok (boolean), color_res (string)
        """
        color_res = ''
        t400 = time.time()
        color_resize_factor = 10
        data = new_csv_reader(newest_csv_path)
        row_count = sum(1 for line in data)
        colors = np.zeros(shape=(row_count, 3))
        for i in range(len(data)):
            ref_color = data[i]
            colors[i] = list(map(int, ref_color[0].split('-')[:3]))
        if len(cropped) != 0:
            ok, color_res = detect_colors_without_mapping(
                cropped, num_clusters=5, num_iters=50, resize_factor=color_resize_factor, crop_factor=50, colors_in_order=colors, csv=data, type="rgb")
        else:
            return 0, ''
        if ok:
            t500 = time.time()
            if DISPLAY_ALL_DELAY == 1:
                print(bcolors.B+"     -->  color detection delay " +
                      str(round(t500-t400, 3)) + ' ('+(str(classe))+')'+bcolors.ENDC)
        return ok, color_res

    # -----------------------------------------------------------------------------------------------------------------
    def manageColor(self, cropped, classe, mappingImages, person, color):
        text_color = ''
        color_res = ''
        bKMEANS = 0
        if DETECT_COLOR == 1:
            if cropped.size != 0:
                bKMEANS, color_res,_ ,mapping = self.colorDetection_withImage(
                    cropped, classe)

                text_color = "--- " + str(color_res)
                color_str = "color: " + str(color_res)
            # --------------------------------------------------
            if bKMEANS == 1 and DISPLAY == 1:
                mappingImages = createImageKMEANS(
                    cropped, color, mappingImages, mapping, color_str, classe)

            # --------------------------------------------------
                if DESCRIBE_PERSON == 1 and person != None and classe == "Human face":
                    person.skin_color = color_res
                elif DESCRIBE_PERSON == 1 and person != None:
                    person.clothes_color = color_res
            # --------------------------------------------------

        return text_color, color_res, mappingImages, person

    ######################################################################
    # services  handler
    ######################################################################

    def handle_ServicePerceptionShoesDrinkSRV(self, shoes_and_drink_detection_service):
        result = self.process_imageNAOQI(
            shoes_and_drink_detection_service.entries_list, publishPerson=0)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def handle_ServicePerceptionSeatSRV(self, object_detection_service):
        result = self.process_imageNAOQI(
            object_detection_service.entries_list, publishPerson=0)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def handle_ServicePerceptionBagSRV(self, object_detection_service):
        result = self.process_imageNAOQI(
            object_detection_service.entries_list, publishPerson=0)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def handle_ServicePerceptionSRV(self, object_detection_service):
        result = self.process_imageNAOQI(
            object_detection_service.entries_list, publishPerson=0)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def handle_ServicePerceptionHumanSRV(self, person_features_detection_service):
        result = self.process_imageNAOQI(
            person_features_detection_service.entries_list, publishPerson=1)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def handle_ServicePerceptionHumanSRVWithDistance(self, person_features_detection_distance_service):
        objectsListRequested = person_features_detection_distance_service.entries_list.obj
        distanceMax = person_features_detection_distance_service.entries_list.distanceMaximum
        result = self.process_imageNAOQI(
            objectsListRequested, publishPerson=1, distanceMaximum=distanceMax)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def handle_ServicePerceptionHumanSRVWithDistancePosture(self, person_features_detection_posture):
        distanceMax = person_features_detection_posture.entries_list.distanceMaximum
        result = self.process_imageNAOQI(
            distanceMaximum=distanceMax, posture=1)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def handle_ServiceWaveHandDetection(self, wave_hand_detection):
        distanceMax = wave_hand_detection.distance_max
        result = self.process_imageNAOQI(distanceMaximum=distanceMax, wave=1)
        return result

    # -----------------------------------------------------------------------------------------------------------------
    def handle_ServicePointingHandDetection(self, pointing_hand_detection):
        distanceMax = pointing_hand_detection.distance_max
        result = self.process_imageNAOQI(distanceMaximum=distanceMax, pointing=1)
        return result


    ######################################################################
    # TRANSFORM
    ######################################################################

    # Get the transform, here we are assuming that the ROS CameraTop_optical_frame uses the same coordinates system as the Naoqi frame

    def initTransformation(self):
        self.transform_camera_map = get_transformation(
            "CameraTop_optical_frame", "odom")

    # -----------------------------------------------------------------------------------------------------------------

    def compute_absolute_pose(self, pose):
        transform = get_transformation("CameraTop_optical_frame", "odom")

        p = Point32()
        p.x = pose[0]
        p.y = pose[1]
        p.z = pose[2]

        new_pose = transform_point(transform, p)

        return new_pose

    ######################################################################
    # MAIN LOOP NAOQI
    ######################################################################

    def convertImage2DPepperToCV(self, pepperImage):
        kYUVColorSpace = 0
        kRGBColorSpace = 11
        kBGRColorSpace = 13
        kYUV422ColorSpace = 9
        kDepthColorSpace = 17
        kRawDepthColorSpace = 23
        img2 = Image()
        img2.header.frame_id = 'camera_top_frame'
        img2.height = pepperImage[1]
        img2.width = pepperImage[0]
        nbLayers = pepperImage[2]
        if pepperImage[3] == kYUVColorSpace:
            encoding = "mono8"
        elif pepperImage[3] == kRGBColorSpace:
            encoding = "rgb8"
        elif pepperImage[3] == kBGRColorSpace:
            encoding = "bgr8"
        elif pepperImage[3] == kYUV422ColorSpace:
            encoding = "yuv422"  # this works only in ROS groovy and later
        elif pepperImage[3] == kDepthColorSpace or pepperImage[3] == kRawDepthColorSpace:
            encoding = "16UC1"
        else:
            rospy.logerr("Received unknown encoding: {0}".format(img2[3]))
        img2.encoding = encoding
        img2.step = img2.width * nbLayers
        img2.data = pepperImage[6]

        return img2

    # -----------------------------------------------------------------------------------------------------------------
    def convertImage3DPepperToCV(self, pepperImageD):
        kDepthColorSpace = 17
        encoding = ""
        img = Image()
        img.header.stamp = rospy.Time.now()
        img.header.frame_id = "camera_depth_frame"
        img.height = pepperImageD[1]
        img.width = pepperImageD[0]
        nbLayers = pepperImageD[2]
        if pepperImageD[3] == kDepthColorSpace:
            encoding = "mono16"
        else:
            rospy.logerr(
                "Received unknown encoding: {0}".format(pepperImageD[3]))
        img.encoding = encoding
        img.step = img.width * nbLayers
        img.data = pepperImageD[6]

        return img

    # -----------------------------------------------------------------------------------------------------------------

    def convertImageBGR8ToCV(self, img2):
        cv_rgb = self.bridge.imgmsg_to_cv2(img2, "bgr8")
        frame = np.array(cv_rgb, dtype=np.uint8)
        if frame.size == 0:
            print(bcolors.WARNING +
                  '     ObjectDetector frame RGB image EMPTY'+bcolors.ENDC)
        t2 = time.time()
        h = frame.shape[0]
        w = frame.shape[1]
        imageBGR = np.array(frame)

        return imageBGR, frame

    # -----------------------------------------------------------------------------------------------------------------

    def process_imageNAOQI(self, objectsRequested=None, publishPerson=0, distanceMaximum=100, wave=0, posture=0, pointing=0):

        # if wave == 1 or posture == 1 or pointing == 1:
        #     rospy.loginfo(
        #         bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] detection in progress ..."+bcolors.ENDC)
        # else:
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision] detection in progress ... "+bcolors.ENDC)

        t_start_images = time.time()
        # -------------------Collect Images -----------------------------
        [pepperImage, pepperImageD] = self.video_service.getImagesRemote(
            self.videosClient)
        if not pepperImage:
            raise Exception("No data in image")

       # ------------------ TIMING  ------------------------------
        t_end_images = time.time()
        if DISPLAY_ALL_DELAY == 1:
            print(bcolors.B+"     --> images acquisition delay " +
                  str(round(t_end_images - t_start_images, 3))+bcolors.ENDC)

        # ------------------- 2D Camera -----------------------------
        img2d = self.convertImage2DPepperToCV(pepperImage)
        imageBGR, frame = self.convertImageBGR8ToCV(img2d)

        # ------------------- 3D Camera -----------------------------
        imgD = self.convertImage3DPepperToCV(pepperImageD)
        depth_image = self.bridge.imgmsg_to_cv2(imgD, "32FC1")

        # ---- posture
        if wave == 1:
            return self.wave_hand_detection(frame, depth_image, distanceMaximum, display=DISPLAY)

        elif pointing == 1:
            return self.pointing_hand_detection(frame, depth_image, distanceMaximum, display=DISPLAY)

        # elif posture == 1:
        #     return self.movenet_multipose(frame, depth_image, distanceMaximum, display=DISPLAY)

        # objects / person
        return self.detectionTask(imageBGR, frame, depth_image, objectsListRequested=objectsRequested, publish_Person=publishPerson, distanceMax=distanceMaximum)

    ######################################################################
    # DEFINE STRATEGY
    ######################################################################

    def defineStrategy(self, objectsListRequested, publish_Person, distanceMax):

        bCOCO = 1
        bOID = 1

        if publish_Person == 1:
            bCOCO = 0

        # --------- PERSON -----
        if publish_Person == 1 and distanceMax == 100:
            print("     ask for Persons detections FEATURES....")

        # --------- PERSON -----
        if publish_Person == 1 and distanceMax != 100:
            print("     ask for Persons detections FEATURES with distance filter....")

        # --------- SEAT -----
        bSeatTakenInfor = 0
        for object in objectsListRequested:
            if "SEAT_INFORMATION" == object.data:
                bSeatTakenInfor = 1
                bOID = 1
                bCOCO = 0
                print("     ask for SEAT_INFORMATION detections ....")

        # --------- BAG -----
        bBagInfor = 0
        for object in objectsListRequested:
            if "BAG_INFORMATION" == object.data:
                bBagInfor = 1
                bOID = 1
                bCOCO = 1
                print("     ask for BAG_INFORMATION detections ....")

        # --------- SHOES and DRINK -----
        bShoesDrinkTakenInfor = 0
        for object in objectsListRequested:
            if "SHOES_DRINK_INFORMATION" == object.data:
                bShoesDrinkTakenInfor = 1
                bOID = 1
                bCOCO = 1
                print("     ask for SHOES_DRINK_INFORMATION detections ....")

        # --------- ALL -----
        bAll = 0
        for object in objectsListRequested:
            if "ALL" == object.data:
                bAll = 1
                print("     ask for ALL detections....")

        if bAll == 0 and bSeatTakenInfor == 0 and publish_Person == 0 and bShoesDrinkTakenInfor == 0 and bBagInfor == 0:
            print("     ask classical detections with resquested objects ....")

        return bCOCO, bOID, bAll, bSeatTakenInfor, bBagInfor, bShoesDrinkTakenInfor


# ----------------------------------------------------------------------------

    def computeDetections(self, cv_rgb, bCOCO, bOID):
        t_start_computing = time.time()
        outputs = None
        outputs_coco = None

        # detect object
        if bOID == 1:
            blob = cv2.dnn.blobFromImage(cv_rgb, size=(
                300, 300), swapRB=False, crop=False)
            self.cvNet.setInput(blob)
            outputs = self.cvNet.forward()

            # ------------------ TIMING  ------------------------------
            t_end_computing = time.time()
            if DISPLAY_ALL_DELAY == 1:
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
            if DISPLAY_ALL_DELAY == 1:
                print(bcolors.B+"     --> computeDetections (COCO) delay " +
                      str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

        return outputs, outputs_coco

# --------------------------------------------------------------
    def matchWithRequestOID(self, outputs_oid, objectsListRequested, bAll, bSeatTakenInfor, bBagInfor, bShoesDrinkTakenInfor, bOID):
        # --------- get only requested Objects -----
        newDetections = []

        # ------ OID v4 -----
        if bOID == 1:
            # print("     OID : ")
            for detection in outputs_oid[0, 0, :, :]:
                score = float(detection[2])
                if score > 0.3:
                    classID = int(detection[1])
                    classe = self.classes[classID - 1]
                    for object in objectsListRequested:
                        # print("       found :"+str(classe)+" request : " + str(object))
                        if classe == object.data:
                            newDetections.append(detection)
                            # print("matching")

        # --------- SEAT -----
        if bSeatTakenInfor == 1:
            newDetections = outputs_oid[0, 0, :, :]

        # --------- BAG -----
        if bBagInfor == 1:
            newDetections = outputs_oid[0, 0, :, :]

        # --------- shoes/drink -----
        if bShoesDrinkTakenInfor == 1:
            newDetections = outputs_oid[0, 0, :, :]

        # --------- ALL -----
        if bAll == 1:
            newDetections = outputs_oid[0, 0, :, :]

        return newDetections


# --------------------------------------------------------------

    def matchWithRequestCOCO(self, outputs_coco, objectsListRequested, bAll, bSeatTakenInfor, bBagInfor, bShoesDrinkTakenInfor, bCOCO):
        newDetections_coco = []

        if bCOCO == 1:
            # print("     COCO 2017 : ")
            for detection_coco in outputs_coco[0, 0, :, :]:
                score_coco = float(detection_coco[2])
                if score_coco > 0.3:
                    classID_coco = int(detection_coco[1])
                    classe_coco = self.classes_coco[classID_coco]
                    for object in objectsListRequested:
                        # print("       found :"+str(classe_coco)+" request : " + str(   object))
                        if classe_coco == object.data:
                            newDetections_coco.append(detection_coco)
                            # print("matching")

                # --------- ALL -----
            if bAll == 1:
                newDetections_coco = outputs_coco[0, 0, :, :]

            # --------- SEAT -----
            if bSeatTakenInfor == 1:
                newDetections_coco = outputs_coco[0, 0, :, :]

            # --------- BAG -----
            if bBagInfor == 1:
                newDetections_coco = outputs_coco[0, 0, :, :]

            # --------- shoes/drink -----
            if bShoesDrinkTakenInfor == 1:
                newDetections_coco = outputs_coco[0, 0, :, :]

        return newDetections_coco

    # ----------------------------------------------------------------------------------------
    def computeModels(self, classe, detection, strOrigine, width, height, frame, score, depth_image, colordisplay, mappingImages, distanceMax, publish_Person):
        left = int(detection[3] * width)
        top = int(detection[4] * height)
        right = int(detection[5] * width)
        bottom = int(detection[6] * height)

        if top < 0:
            top = 0
        if left < 0:
            left = 0
        if right < 0:
            right = 0
        if bottom < 0:
            bottom = 0
        # left,top ---
        # |          |
        # |          |
        # |          |
        # --------right,bottom

        person = None

        if DETECT_AGE == 1 or DETECT_COLOR == 1:
            cropped = frame[int(top):int(bottom), int(
                left):int(right)]   # [y:y+h, x:x+h]

        ############## human ##############
        if DESCRIBE_PERSON == 1:
            person = self.createPerson(classe)

        ############## age ##############
        text_age = ''
        if DETECT_AGE == 1:
            text_age = self.manageAgeOID(cropped, classe)

        ############## color ##############
        text_color = ''
        color_res = ''
        if DETECT_COLOR == 1:
            if DISPLAY == 1:
                text_color, color_res, mappingImages, person = self.manageColor(
                    cropped, classe, mappingImages, person, colordisplay)
            else:
                ok, color_res = self.colorDetection(
                    cropped, classe)
                text_color = "--- " + str(color_res)

        ############## distance ##############
        dist, point_x, point_y, point_z = 0, 0, 0, 0
        absolute_coord = [0, 0, 0]

        if DETECT_DISTANCE == 1:
            if depth_image.size == 0:
                print(bcolors.WARNING +
                      '     depth_image image  EMPTY'+bcolors.ENDC)

            else:
                dist, point_x, point_y, point_z, Xcenter, Ycenter = detectDistanceResolution(
                    depth_image, left, bottom, top, right, [640, 480])
                absolute_coord = self.compute_absolute_pose(
                    [point_x, point_y, point_z])
                if DESCRIBE_PERSON == 1 and person != None:
                    person.distance = dist

        ############## describe person ##############
        if DESCRIBE_PERSON == 1 and person != None and publish_Person == 1:
            if float(person.distance) < distanceMax:
                person.coord = Point32(
                    absolute_coord[0], absolute_coord[1], absolute_coord[2])
                self.personsDetectedMANAGER.person_list.append(
                    person)
            else:
                print(bcolors.WARNING+'     person detected but out of distanceMax : ' +
                      str(distanceMax) + '  / distance : ' + str(person.distance) + bcolors.ENDC)

            person = None

        ############## store for MANAGER ##############
        if float(dist) < distanceMax:
            obj = Object()
            obj.label.data = classe
            obj.coord = Point32(
                absolute_coord[0], absolute_coord[1], absolute_coord[2])
            obj.distance = float(dist)
            obj.score = score
            obj.color.data = str(color_res)

            obj.height_img = bottom-top  # for RViz scaling
            obj.width_img = right - left
            obj.bounding_box.top_left.x = left
            obj.bounding_box.top_left.y = top
            obj.bounding_box.bottom_right.x = right
            obj.bounding_box.bottom_right.y = bottom

            self.objectsDetectedMANAGER.object_list.append(obj)
        else:
            print(bcolors.WARNING+'     object detected but out of distanceMax : ' +
                  str(distanceMax) + '  / distance : ' + str(dist) + bcolors.ENDC)

        ############## display ##############

        if DISPLAY == 1:
            manageDisplay(score, left, top, right, bottom, frame, depth_image,
                          dist, classe, absolute_coord[0], absolute_coord[1], absolute_coord[2], colordisplay)

        ############## end display ##############
        print('     -->  ' + str(classe) + ' (' + str(int(score*100)) + '%) --- ' + str(
            format(dist, '.2f')) + ' m' + str(text_age) + ' ' + str(text_color) + ' (from ' + strOrigine + ')')

    # ---------------------------------------------------------------------------------------
    def adjustCoco(self, bOID, bCOCO, newDetections_coco, newDetections_oid, width, height):

        newDetections2_coco = []
        if bCOCO == 1:
            if bOID == 1:
                newDetections2_coco = clean_coco_oid( self.classes_coco, newDetections_coco, self.classes, newDetections_oid, width, height)
            else:
                newDetections2_coco = newDetections_coco

        return newDetections2_coco

    # ---------------------------------------------------------------------------------------
    def adjustOID(self, bOID, bCOCO, newDetections_coco, newDetections_oid, width, height):

        newDetections2_oid = []
        if bOID == 1:
            if bCOCO == 1:
                newDetections2_oid = clean_oid_coco(
                    self.classes_coco, newDetections_coco, self.classes, newDetections_oid, width, height)
            else:
                newDetections2_oid = newDetections_oid

        return newDetections2_oid

    # ---------------------------------------------------------------------------------------
    def computeModelsOnAllCOCO(self, newDetections2_coco, width, height, frame, depth_image, mappingImages, distanceMax, publish_Person):

        print('     COCO : ')
        for detection_coco in newDetections2_coco:
            score = float(detection_coco[2])
            if score > 0.3:
                classID_coco = int(detection_coco[1])
                classe_coco = self.classes_coco[classID_coco]
                self.computeModels(classe_coco, detection_coco, "COCO", width, height, frame, score,
                                   depth_image, self.colors_coco[classID_coco], mappingImages, distanceMax, publish_Person)

    # ---------------------------------------------------------------------------------------
    def computeModelsOnAllOID(self, newDetections2_oid, width, height, frame, depth_image, mappingImages, distanceMax, publish_Person):

        print('     OID : ')
        for detection in newDetections2_oid:
            bKNN = False
            score = float(detection[2])
            if score > 0.3:
                classID = int(detection[1])
                classe = self.classes[classID - 1]
                self.computeModels(classe, detection, "OID", width, height, frame, score, depth_image,
                                   self.colors[classID - 1], mappingImages, distanceMax, publish_Person)

    # ---------------------------------------------------------------------------------------
    def manageSeatTaken(self, bSeatTakenInfor, bOID, bCOCO, newDetections2_oid, newDetections2_coco, width, height, depth_image):

        if bSeatTakenInfor == 1:
            t_start_computing = time.time()

            print("     SEAT info : ")
            self.objectsDetectedMANAGER = ObjectList()

            arr_empty_chairs = []
            arr_taken_chairs = []
            arr_persons = []
            if bOID == 1:
                arr_persons, arr_empty_chairs, arr_taken_chairs = has_chairs_couch_oid(
                    self.classes, newDetections2_oid, width, height)
            elif bCOCO == 1:
                arr_persons, arr_empty_chairs, arr_taken_chairs = has_chairs_coco_couch(
                    self.classes_coco, newDetections2_coco, width, height)

            for chair in arr_empty_chairs:
                if depth_image.size == 0:
                    print(bcolors.WARNING +
                          '     depth_image image  EMPTY'+bcolors.ENDC)
                else:
                    # chair.xyxy = [top, left, bottom, right]
                    top = chair.xyxy[0]
                    left = chair.xyxy[1]
                    bottom = chair.xyxy[2]
                    right = chair.xyxy[3]
                    dist, point_x, point_y, point_z, Xcenter, Ycenter = detectDistanceResolution(
                        depth_image, left, bottom, top, right, [640, 480])
                    absolute_coord = self.compute_absolute_pose(
                        [point_x, point_y, point_z])

                    ############## store for MANAGER ##############
                    obj = Object()
                    obj.label.data = "Empty Seat"
                    obj.coord = Point32(
                        absolute_coord[0], absolute_coord[1], absolute_coord[2])
                    obj.distance = float(dist)
                    obj.score = .5
                    obj.color.data = ''

                    obj.height_img = bottom-top  # for Rviz scaling
                    obj.width_img = right-left

                    self.objectsDetectedMANAGER.object_list.append(obj)

            # ------------------ TIMING  ------------------------------
            t_end_computing = time.time()
            if DISPLAY_ALL_DELAY == 1:
                print(bcolors.B+"     --> seat taken delay " +
                      str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

    # ---------------------------------------------------------------------------------------
    def manageBag(self, bBagInfor, bOID, bCOCO, newDetections2_oid, newDetections2_coco, width, height, depth_image):

        if bBagInfor == 1:
            t_start_computing = time.time()

            print("     BAG info : ")
            self.objectsDetectedMANAGER = ObjectList()

            arr_bag = []

            if bOID == 1 and bCOCO == 1:
                arr_bag = has_bag_oid_coco(self.classes, newDetections2_oid,
                                           self.classes_coco, newDetections2_coco, width, height)

            for bag in arr_bag:
                if depth_image.size == 0:
                    print(bcolors.WARNING +
                          '     depth_image image  EMPTY'+bcolors.ENDC)
                else:
                    # chair.xyxy = [top, left, bottom, right]
                    top = bag[0]
                    left = bag[1]
                    bottom = bag[2]
                    right = bag[3]
                    dist, point_x, point_y, point_z, Xcenter, Ycenter = detectDistanceResolution(
                        depth_image, left, bottom, top, right, [640, 480])
                    absolute_coord = self.compute_absolute_pose(
                        [point_x, point_y, point_z])

                    print("    --> found a bag ")
                    ############## store for MANAGER ##############
                    obj = Object()
                    obj.label.data = "Bag"
                    obj.coord = Point32(
                        absolute_coord[0], absolute_coord[1], absolute_coord[2])
                    obj.distance = float(dist)
                    obj.score = .5
                    obj.color.data = ''

                    obj.height_img = bottom-top  # for Rviz scaling
                    obj.width_img = right-left
                    obj.bounding_box.top_left.x = left
                    obj.bounding_box.top_left.y = top
                    obj.bounding_box.bottom_right.x = right
                    obj.bounding_box.bottom_right.y = bottom

                    self.objectsDetectedMANAGER.object_list.append(obj)

            # ------------------ TIMING  ------------------------------
            t_end_computing = time.time()
            if DISPLAY_ALL_DELAY == 1:
                print(bcolors.B+"     --> bag delay " +
                      str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

    # ------------------------------------------------------------------------------------------------------------
    def manageShoesDrink(self, bShoesDrinkTakenInfor, newDetections2_oid, newDetections2_coco, width, height):

        if bShoesDrinkTakenInfor == 1:

            # ------------------ TIMING  ------------------------------
            t_start_computing = time.time()

            print("     SHOES / DRINK info : ")
            messageList = []

            result_dictionnaire = has_shoes_on_drink_oid_coco(
                self.classes, newDetections2_oid, self.classes_coco, newDetections2_coco, width, height)
            for text, positionYhuman in result_dictionnaire.items():
                messageList.append(str(text))

            mess = shoes_and_drink_detection_serviceResponse()
            mess.outputs_list_broken_rules = messageList

            # ------------------ TIMING  ------------------------------
            t_end_computing = time.time()
            if DISPLAY_ALL_DELAY == 1:
                print(bcolors.B+"     --> shoes and drink detection delay " +
                      str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

            return mess

    # ---------------------------------------------------------------------------------------
    def create_text_marker(self, obj, id, DURATION):
        """
        Create ros rviz text marker message
        """
        text_marker = Marker()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.id = id
        text_marker.header.frame_id = 'odom'
        text_marker.header.stamp = rospy.get_rostime()
        text_marker.action = text_marker.ADD

        # object class as the text of the marker
        text_marker.text = obj.label.data

        # blue text of object label
        text_marker.color.r = 0.0
        text_marker.color.g = 0.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0

        # text marker with fixed scale
        text_marker.scale.x = 0.4
        text_marker.scale.y = 0.4
        text_marker.scale.z = 0.06

        # location
        # can't set two marker (obj marker and its label marker) at the same xyz, otherwise it does not show
        coord_offset = 0.01
        text_marker.pose.position.x = obj.coord.x + coord_offset
        text_marker.pose.position.y = obj.coord.y
        text_marker.pose.position.z = obj.coord.z

        # Marker Life-time
        text_marker.lifetime = rospy.Duration(DURATION)

        return text_marker

    # ---------------------------------------------------------------------------------------
    def create_obj_marker(self, obj, id, DURATION):
        # Get Obj Info
        obj_coord = obj.coord  # is Point32() on odom frame
        od_to_cam_tf = get_transformation("odom", "CameraTop_optical_frame")
        cam_pose = transform_point(od_to_cam_tf, obj_coord)
        cam_dist = cam_pose[2]  # cam optical frame z
        bbx_width = obj.width_img
        bbx_height = obj.height_img

        # Initialize Marker
        marker = Marker()
        marker.id = id
        marker.header.frame_id = 'odom'
        marker.header.stamp = rospy.get_rostime()
        marker.action = marker.ADD

        # ------------------ Marker Representations  ------------------------------
        # use cylinder to represent human and sphere to represent objects detected
        classe = obj.label.data
        human_type_lst = ['Man', 'Boy', 'Girl', 'Woman', 'person']

        if classe in human_type_lst:  # person: red cylinder
            marker.type = Marker.CYLINDER

            # Set the color as red
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5

        else:  # non-human object: green shpere
            marker.type = Marker.SPHERE

            # Set the color as green
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5

        # ------------------ Marker Scale  ------------------------------
        m_fx = 525
        m_fy = 525
        # pinhole model of a camera
        world_width = (cam_dist/m_fx)*bbx_width
        world_height = (cam_dist/m_fy)*bbx_height

        marker.scale.x = 0.3  # set the obj world z scale to be 0.3
        marker.scale.y = world_width
        marker.scale.z = world_height

        # ------------------ Marker Pose  ------------------------------
        # position
        marker.pose.position.x = obj_coord.x
        marker.pose.position.y = obj_coord.y
        marker.pose.position.z = obj_coord.z

        # orientation
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Marker Lifetime
        marker.lifetime = rospy.Duration(DURATION)
        return marker

    # ---------------------------------------------------------------------------------------
    def show_RViz_marker_arr(self, object_list, DURATION=50):
        marker_lst = MarkerArray()

        for i, obj in enumerate(object_list):
            # graphical marker id: 0, 2, 4,..
            obj_marker_id = i*2

            # text label id: 1, 3, 5,..
            text_marker_id = 1 + i*2

            # Cteate text and object markers
            marker = self.create_obj_marker(obj, obj_marker_id, DURATION)
            text_marker = self.create_text_marker(obj, text_marker_id, DURATION)

            # added to the marker array
            marker_lst.markers.append(marker)
            marker_lst.markers.append(text_marker)

        self.marker_arr_pub.publish(marker_lst)
        return

    ######################################################################
    # MAIN LOOP ROS or NAOQI
    ######################################################################

    def detectionTask(self, cv_rgb, frame, depth_image, objectsListRequested, publish_Person=0, distanceMax=100):

        mappingImages = []

        t_start_computing = time.time()

        self.objectsDetectedMANAGER = ObjectList()
        self.personsDetectedMANAGER = PersonList()

        width = cv_rgb.shape[1]
        height = cv_rgb.shape[0]

        # ----------------------------------
        bCOCO, bOID, bAll, bSeatTakenInfor, bBagInfor, bShoesDrinkTakenInfor = self.defineStrategy(
            objectsListRequested, publish_Person, distanceMax)

        outputs_oid, outputs_coco = self.computeDetections(cv_rgb, bCOCO, bOID)

        # get objects according to request
        newDetections_oid = self.matchWithRequestOID(
            outputs_oid, objectsListRequested, bAll, bSeatTakenInfor, bBagInfor, bShoesDrinkTakenInfor, bOID)
        newDetections_coco = self.matchWithRequestCOCO(
            outputs_coco, objectsListRequested, bAll, bSeatTakenInfor, bBagInfor, bShoesDrinkTakenInfor, bCOCO)

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
        self.manageSeatTaken(bSeatTakenInfor, bOID, bCOCO, newDetections2_oid,
                             newDetections2_coco, width, height, depth_image)
        mess = self.manageShoesDrink(
            bShoesDrinkTakenInfor, newDetections2_oid, newDetections2_coco, width, height)
        self.manageBag(bBagInfor, bOID, bCOCO, newDetections2_oid,
                       newDetections2_coco, width, height, depth_image)

        # ###### publish Image with detection + distance + color ################
        if DISPLAY == 1:
            t_start_publish = time.time()
            self.publishDisplayImage(frame, depth_image)
            if DETECT_COLOR == 1:
                self.publishDisplayColor(mappingImages)
            t_end_publish = time.time()
            if DISPLAY_ALL_DELAY == 1:
                rospy.loginfo(bcolors.B+"     --> publish images delay " +
                      str(round(t_end_publish - t_start_publish, 3))+bcolors.ENDC)

       # ------------------ TIMING  ------------------------------
        t_end_computing = time.time()
        if DISPLAY_DELAY == 1:
            rospy.loginfo(bcolors.B+"     --> TOTAL detection delay " +
                  str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision] detection done "+bcolors.ENDC)

        # ------------------ RViz Marker ------------------------------
        if self.objectsDetectedMANAGER.object_list and DISPLAY == 1:
            # print(self.objectsDetectedMANAGER.object_list)
            self.show_RViz_marker_arr(self.objectsDetectedMANAGER.object_list, DURATION=50)

        ###### Service ################

        if (publish_Person == 1):
            return self.personsDetectedMANAGER
        elif bShoesDrinkTakenInfor == 1:
            return mess
        else:
            return self.objectsDetectedMANAGER

    ############ WHAT TO DO AT THE END #################
    def cleanup(self):
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision] Shutting down vision node"+bcolors.ENDC)
        self.video_service.unsubscribe(self.videosClient)
        self.session.close()

    ######################################################################
    # Posture
    ######################################################################
  # -----------------------------------------------------------------------------------------------------------------

    # def movenet_multipose(self, image, depth_data, distanceMax, display=False):
    #     print("     ask for pose detection ....")
    #     time_1 = time.time()

    #     list_persons = self.pose_detector.detect(image)
    #     final_msg_pose : List[PersonPose()] = []
    #     final_msg_features : List[Person()] = []

    #     if len(list_persons) == 0:
    #         print("     no persons detected ....")
    #         rospy.loginfo(
    #             bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] detection done..."+bcolors.ENDC)
    #         return {"outputs_list": final_msg_features, "outputs_pose_list": final_msg_pose}

    #     for person in list_persons:
    #         if person.score < 0.3:
    #             pass
    #         else:
    #             print("     -->  Person detected")
    #             person_pose = Person_pose()
    #             person_pose.posture.data = "standing"
    #             person_pose.height = float(0.0)

    #             start_x = person.bounding_box.start_point.x
    #             start_y = person.bounding_box.start_point.y
    #             end_x = person.bounding_box.end_point.x
    #             end_y = person.bounding_box.end_point.y

    #             dist, point_x, point_y, point_z, Xcenter, Ycenter = detectDistanceResolution(
    #                 depth_data, start_x, end_y, start_y, end_x, [640, 480])
    #             if dist > distanceMax:
    #                 continue

    #             if self.pose_detection.is_waving(person):
    #                 person_pose.posture.data = "waving"

    #             if self.pose_detection.is_sit_down(person):
    #                 person_pose.posture.data = "sit down"

    #             height = float(
    #                 self.pose_detection.get_height(person, depth_data))

    #             if height != False:
    #                 person_pose.height = height

    #             # INPUTS: image,  start_x, start_y, end_x, end_y
    #             # OUTPUTS: age,
    #             ######################################################################################

    #             cropped = image[int(start_y):int(end_y), int(
    #                 start_x):int(end_x)]   # [y:y+h, x:x+h]
    #             age, gender, face_cropped = self.age_gender_detect.detect(
    #                 cropped)
    #             person = Person()
    #             person.name = String("")
    #             person.clothes_color = String("")
    #             person.age = String("")
    #             person.gender = String("")
    #             person.skin_color.data = "White"
    #             person.distance = 0.0
    #             person.coord.x = 0
    #             person.coord.y = 0
    #             person.coord.z = 0

    #             if age is not None:
    #                 person.age = String(age)
    #             if gender is not None:
    #                 person.gender = String(gender)

    #             ############## color ##############
    #             ok, clothes_color_label = self.colorDetection(
    #                 cropped, "Person Multipose")
    #             if ok:
    #                 person.clothes_color = String(clothes_color_label)

    #             if np.any(face_cropped):
    #                 person.skin_color = String("White")

    #             ############## person ##############

    #             coordtmp = self.compute_absolute_pose(
    #                 [point_x, point_y, point_z])

    #             person.distance = float(dist)
    #             person.coord.x = float(coordtmp[0])
    #             person.coord.y = float(coordtmp[1])
    #             person.coord.z = float(coordtmp[2])

    #             ######################################################################################

    #             final_msg_pose.append(person_pose)
    #             final_msg_features.append(person)

    #     # display = True
    #     if display:
    #         frame, dict_wave_person = visualize(image, list_persons)
    #         visu_message = self.bridge.cv2_to_imgmsg(frame, "bgr8")
    #         self.pub_visu.publish(visu_message)

    #     time_2 = time.time()

    #     if DISPLAY_ALL_DELAY == 1:
    #         print(bcolors.B+"     -->  Pose/Vision detection delay " +
    #               str(round(time_2-time_1, 3)) + bcolors.ENDC)

    #     print("        -->  Number of poses detected: ",
    #           len(final_msg_pose))
    #     print("        -->  Number of persons detected: ",
    #           len(final_msg_features))

    #     rospy.loginfo(
    #         bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] detection done..."+bcolors.ENDC)

    #     return {"outputs_list": final_msg_features, "outputs_pose_list": final_msg_pose}

    # -----------------------------------------------------------------------------------------------------------------

#     def wave_hand_detection(self, image, depth_data, distanceMax, display=False):
#         print("     ask for wave hand detection ....")

#         time_1 = time.time()

#         list_persons = self.pose_detector.detect(image)
#         final_msg_pose = PoseArray()
#         h = std_msgs.msg.Header()
#         h.stamp = rospy.Time.now()
#         h.frame_id = "odom"
#         final_msg_pose.header = h

#         if len(list_persons) == 0:
#             print("     no persons detected ....")
#             rospy.loginfo(
#                 bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] detection done..."+bcolors.ENDC)
#             return final_msg_pose

#         for person in list_persons:
#             if person.score < 0.3:
#                 pass
#             else:
#                 print("     -->  Person detected")
#                 start_x = person.bounding_box.start_point.x
#                 start_y = person.bounding_box.start_point.y
#                 end_x = person.bounding_box.end_point.x
#                 end_y = person.bounding_box.end_point.y

#                 dist, point_x, point_y, point_z, Xcenter, Ycenter = detectDistanceResolution(
#                     depth_data, start_x, end_y, start_y, end_x, [640, 480])

#                 if dist > distanceMax:
#                     continue
#                 if self.pose_detection.is_waving(person):
#                     coord_odom = self.compute_absolute_pose(
#                         [point_x, point_y, point_z])

#                     person_pose = Pose()
#                     person_pose.position.x = coord_odom[0]
#                     person_pose.position.y = coord_odom[1]
#                     person_pose.position.z = coord_odom[2]

#                     final_msg_pose.append(person_pose)

#         if display:
#             frame, dict_wave_person = visualize(image, list_persons)
#             visu_message = self.bridge.cv2_to_imgmsg(frame, "bgr8")

#             self.pub_visu.publish(visu_message)

#         time_2 = time.time()
#         if DISPLAY_ALL_DELAY == 1:
#             print(bcolors.B+"     -->  waving hand delay " +
#                   str(round(time_2-time_1, 3)) + bcolors.ENDC)

#         print("        -->  Number of waving hand detected: ",
#               len(final_msg_pose))

#         rospy.loginfo(
#             bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] detection done..."+bcolors.ENDC)

#         return final_msg_pose
#  # -----------------------------------------------------------------------------------------------------------------

#     def pointing_hand_detection(self, image, depth_data, distanceMax, display=False):
#         print("     ask for pointing hand detection ....")

#         time_1 = time.time()

#         list_persons = self.pose_detector.detect(image)
#         rightList = []
#         topList = []

#         if len(list_persons) == 0:
#             print("     no persons detected ....")
#             rospy.loginfo(
#                 bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] detection done..."+bcolors.ENDC)
#             return {"right_list": rightList, "top_list": topList}

#         for person in list_persons:
#             if person.score < 0.3:
#                 pass
#             else:
#                 print("     -->  Person detected")
#                 start_x = person.bounding_box.start_point.x
#                 start_y = person.bounding_box.start_point.y
#                 end_x = person.bounding_box.end_point.x
#                 end_y = person.bounding_box.end_point.y

#                 dist, point_x, point_y, point_z, Xcenter, Ycenter = detectDistanceResolution(
#                     depth_data, start_x, end_y, start_y, end_x, [640, 480])

#                 if dist > distanceMax:
#                     continue

#                 bPointing, bRight, bTop = self.pose_detection.is_pointing(person)

#                 if bPointing:
#                     rightList.append(bRight)
#                     topList.append(bTop)

#         if display:
#             frame, dict_wave_person = visualize(image, list_persons)
#             visu_message = self.bridge.cv2_to_imgmsg(frame, "bgr8")

#             self.pub_visu.publish(visu_message)

#         time_2 = time.time()
#         if DISPLAY_ALL_DELAY == 1:
#             print(bcolors.B+"     -->  pointing hand delay " +
#                   str(round(time_2-time_1, 3)) + bcolors.ENDC)

#         rospy.loginfo(
#             bcolors.ON_PURPLE+"[RoboBreizh - Pose/Vision] detection done..."+bcolors.ENDC)

#         return {"right_list": rightList, "top_list": topList}


#########################################################################
#########################################################################
if __name__ == '__main__':

    rospy.init_node('object_detection_node_test', anonymous=True)

    DESCRIBE_PERSON = rospy.get_param('~DESCRIBE_PERSON')
    DETECT_AGE = rospy.get_param('~DETECT_AGE')
    DETECT_COLOR = rospy.get_param('~DETECT_COLOR')
    DETECT_DISTANCE = rospy.get_param('~DETECT_DISTANCE')
    DISPLAY_DELAY = rospy.get_param('~DISPLAY_DELAY')
    DISPLAY_ALL_DELAY = rospy.get_param('~DISPLAY_ALL_DELAY')
    DISPLAY = rospy.get_param('~DISPLAY')

    POSTURE_PERSON = rospy.get_param('~posture')
    WAVE_HAND = rospy.get_param('~wave')
    POINTING_HAND = rospy.get_param('~pointing')

    session = qi.Session()
    try:
        # session.connect("tcp://127.0.0.1:9559")
        session.connect("tcp://192.168.50.44:9559")
    except RuntimeError:
        print(("[RoboBreizh - Vision]  Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) + ".\n"
               "Please check your script arguments. Run with -h option for help."))
        sys.exit(1)

    RobobreizhCV_Detector(session)
