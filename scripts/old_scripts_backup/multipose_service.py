#!/tmp/gentoo/usr/bin/env python3
from cv_bridge import CvBridge
import cv2
import os
import time
import sys
import numpy as np
import rospkg
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point32, PointStamped
from perception_utils.utils import *
from robobreizh_msgs.msg import PersonPose,PersonPoseList
from robobreizh_msgs.srv import multipose_service
from PoseDetection.MoveNet_MultiPose.movenet_multipose import MoveNetMultiPose
from std_msgs.msg import String
import std_msgs
# import ml_pose
import tf2_ros
import tf2_geometry_msgs
import math
import perception_utils.roboBreizh_Utils_pose
from AgeDetection.age_gender_detection import AgeGender
from tflite_runtime.interpreter import Interpreter
# NAOQI
import qi

#########################################################################################
# TERMINAL color
#########################################################################################
W = '\033[0m'  # white (normal)
R = '\033[31m'  # red
G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue
PURPLE = '\033[35m'  # purple
CYAN = '\033[96m'
ON_PURPLE = '\033[45m'      # Background Purple

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (147, 20, 255),
    (0, 2): (255, 255, 0),
    (1, 3): (147, 20, 255),
    (2, 4): (255, 255, 0),
    (0, 5): (147, 20, 255),
    (0, 6): (255, 255, 0),
    (5, 7): (147, 20, 255),
    (7, 9): (147, 20, 255),
    (6, 8): (255, 255, 0),
    (8, 10): (255, 255, 0),
    (5, 6): (0, 255, 255),
    (5, 11): (147, 20, 255),
    (6, 12): (255, 255, 0),
    (11, 12): (0, 255, 255),
    (11, 13): (147, 20, 255),
    (13, 15): (147, 20, 255),
    (12, 14): (255, 255, 0),
    (14, 16): (255, 255, 0)
}

body_part_list = ['Nose', 'Left eye', 'Right eye', 'Left ear', 'Right ear', 'Left shoulder', 'Right shoulder', 'Left elbow',
                  'Right elbow', 'Left wrist', 'Right wrist', 'Left hip', 'Right hip', 'Left knee', 'Right knee', 'Left ankle', 'Right ankle']


class Pose_Detector():
    def __init__(self, session, transform=None, standalone=False):
        base_path = get_pkg_path()

        # here is the path of keras_model.tflite
        path = os.path.join(
            base_path, "scripts/models/tensorflow_lite_sitting/keras_model.tflite")
        interpreter = Interpreter(model_path=path)

        self._input_details = interpreter.get_input_details()
        self._output_details = interpreter.get_output_details()
        # self._input_type = self._input_details[0]['dtype']

        self._interpreter_sitdown = interpreter
        if standalone:
            rospy.init_node('person_pose_node', anonymous=True)
            self.bridge = CvBridge()
            self.session = session
            tf_cache_duration = 5.0
            self.camera_dim = [640, 480]

            self.tf_buffer = tf2_ros.Buffer(rospy.Duration(tf_cache_duration))
            tf2_ros.TransformListener(self.tf_buffer)

            self.transform_camera_map = self.get_transformation(
                "CameraTop_optical_frame", "odom")
            self.age_gender_detect = AgeGender()
            self.model = 'movenet_lightning'
            self.tracker = 'bounding_box'
            self.initCamerasNaoQi()
            # Movenet Lightning init
            # rp = rospkg.RosPack()
            # package_path = rp.get_path('perception_pepper')
            # models_path = os.path.join(package_path, 'scripts/models/movenet_lightning/movenet_lightning.tflite')
            self.output_publisher_naoqi = rospy.Publisher(
                "/roboBreizh_detector/perception_pepper/tracking_object", PointStamped, queue_size=10)
            self.output_publisher_naoqi2 = rospy.Publisher(
                "/roboBreizh_detector/perception_pepper/tracking_object_2", PointStamped, queue_size=10)

            self.pub_visu = rospy.Publisher(
                '/roboBreizh_detector/multipose', Image, queue_size=1)

            # Initialize the pose estimator selected.

            rospy.loginfo(
                ON_PURPLE+"[RoboBreizh - Pose detection] Pose detection ... "+W)

            self.pose_detector = MoveNetMultiPose("movenet_multipose", None)

            s = rospy.Service('/robobreizh/perception_pepper/multipose_service',
                              multipose_service, self.handle_ServicePerceptionSRV)

            # spin
            print("Waiting for image topics...")
            rospy.on_shutdown(self.cleanup)		# What we do during shutdown
            rospy.spin()
        else:
            self.camera_dim = [640, 480]

            self.transform_camera_map = transform
            self.session = session
            tf_cache_duration = 5.0

            self.tf_buffer = tf2_ros.Buffer(rospy.Duration(tf_cache_duration))
            tf2_ros.TransformListener(self.tf_buffer)

            self.session = session
            self.output_publisher_naoqi = rospy.Publisher(
                "/roboBreizh_detector/perception_pepper/tracking_object", PointStamped, queue_size=10)
            self.output_publisher_naoqi2 = rospy.Publisher(
                "/roboBreizh_detector/perception_pepper/tracking_object_2", PointStamped, queue_size=10)

    ######################################################################
    # NAOQI CAM
    ######################################################################

    def initCamerasNaoQi(self):
        self.video_service = session.service("ALVideoDevice")
        fps = 2
        resolutionD = 1  	# Image of 320*240px
        colorSpaceD = 17  	# mono16
        resolution = 1  	# Image of 320*240px
        colorSpace = 11  	# RGB

        self.videosClient = self.video_service.subscribeCameras(
            "cameras_pepper", [0, 2], [resolution, resolutionD], [colorSpace, colorSpaceD], fps)

        # periodic task
        # rospy.Timer(rospy.Duration(0.5), self.process_imageNAOQI)

    def handle_ServicePerceptionSRV(self, multipose_service):
        result = self.process_imageNAOQI(multipose_service.entries_list)
        return result

    def process_imageNAOQI(self, entries_list):
        time_1 = time.time()
        rospy.loginfo(
            ON_PURPLE+"[RoboBreizh - Pose detection] detection in progress ... "+W)

        # ------------------------------------------------
        [pepperImage, pepperImageD] = self.video_service.getImagesRemote(
            self.videosClient)
        if not pepperImage:
            raise Exception("No data in image")

        # ------------------- 2D Camera -----------------------------
        kYUVColorSpace = 0
        kRGBColorSpace = 11
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

        cv_rgb = self.bridge.imgmsg_to_cv2(img2, "bgr8")
        frame = np.array(cv_rgb, dtype=np.uint8)
        if frame.size == 0:
            print(O+'     ObjectDetector frame RGB image EMPTY'+W)

        # ------------------- 3D Camera -----------------------------
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

        depth_data = self.bridge.imgmsg_to_cv2(img, "32FC1")

        image = cv2.flip(cv_rgb, 1)

        list_persons = self.pose_detector.detect(image)
        final_msg = PersonPoseList()
        final_msg.person_pose_list = []

        for person in list_persons:
            if person.score < 0.3:
                return False
            else:

                print("     -->  Person detected ")
                person_pose = PersonPose()
                person_pose.posture.data = "standing"
                person_pose.height = float(0.0)

                if self.is_waving(person):
                    person_pose.posture.data = "waving"

                if self.is_sit_down(person):
                    person_pose.posture.data = "sit down"

                height = float(self.get_height(person, depth_data))

                if height != False:
                    person_pose.height = height

                start_x = person.bounding_box.start_point.x
                start_y = person.bounding_box.start_point.y
                end_x = person.bounding_box.end_point.x
                end_y = person.bounding_box.end_point.y

                cropped = image[int(start_y):int(end_y), int(
                    start_x):int(end_x)]   # [y:y+h, x:x+w]
                res_image = self.age_gender_detect.detect(cropped)

                if res_image:
                    visu_message = self.bridge.cv2_to_imgmsg(res_image, "bgr8")
                    self.pub_visu.publish(visu_message)

                dist, point_x, point_y, point_z, Xcenter, Ycenter = detectDistanceResolution(
                    depth_data, start_x, end_y, start_y, end_x, [640, 480])

                coordtmp = self.compute_absolute_pose(
                    [point_x, point_y, point_z])
                # person_pose.coord = Point32()
                # person_pose.coord.x = coordtmp[0]
                # person_pose.coord.y = coordtmp[1]
                # person_pose.coord.z = coordtmp[2]

                final_msg.append(person_pose)
                visu_msg = PointStamped()
                h = std_msgs.msg.Header()
                h.stamp = rospy.Time.now()
                h.frame_id = "odom"
                visu_msg.header = h
                visu_msg.point.x = coordtmp[0]
                visu_msg.point.y = coordtmp[1]
                visu_msg.point.z = coordtmp[2]

                self.output_publisher_naoqi.publish(visu_msg)

        # frame, dict_wave_person = ml_pose.visualize(image, list_persons)
        # visu_message = self.bridge.cv2_to_imgmsg(frame, "bgr8")

        # self.pub_visu.publish(visu_message)

        time_2 = time.time()

        # print("Processing Time: " + str(time_2-time_1))
        # print(final_msg)
        rospy.loginfo(
            ON_PURPLE+"[RoboBreizh - Pose detection] detection done ... "+W)

        return final_msg

    def get_height(self, person, depth_image):
        keypoints = person.keypoints
        keypoint_threshold = 0.05

        head = ['Left eye', 'Right eye', 'Left ear', 'Right ear']
        for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (keypoints[edge_pair[0]].score > keypoint_threshold and
                    keypoints[edge_pair[1]].score > keypoint_threshold):
                for pose in keypoints:
                    if body_part_list[pose[0].value] in head:
                        if self.check_outside_frame(pose[1]):
                            return -1.0
                        else:
                            # print("coord gaze: ",[pose[1].x, pose[1].y])
                            resolutionD_height = 240
                            resolutionD_width = 320
                            resolutionRGB_height = 480
                            resolutionRGB_width = 640

                            ratio_height = resolutionD_height / resolutionRGB_height
                            ratio_width = resolutionD_width / resolutionRGB_width

                            value = depth_image.item(
                                int(pose[1].y * ratio_height), int(pose[1].x * ratio_width))

                            # print("value depth ", value)
                            if value == 0:
                                pass

                            # get the distance of that point (Z coordinate)

                            # ----- 640 x 480 / Cam D ----
                            #  cam_info_msg.K = boost::array<double, 9>{{ 525, 0, 319.5000000, 0, 525, 239.5000000000000, 0, 0, 1  }};
                            m_fx = 525
                            m_fy = 525
                            m_cx = 319.5
                            m_cy = 239.5

                            # pinhole model of a camera
                            point_z = value * 0.001
                            point_x = - (((pose[1].x-m_cx) * point_z) / m_fx)
                            point_y = - (((pose[1].y-m_cy) * point_z) / m_fy)

                            # dist = math.sqrt(point_x * point_x + point_y * point_y + point_z * point_z)
                            # print("point gaze: ", [point_x, point_y, point_z])
                            coord = self.compute_absolute_pose(
                                [point_x, point_y, point_z])
                            # print("coord gaze frame odom? ", [coord[0], coord[1], coord[2]])

                            # coord eyes or ears + 10cm for global height
                            return coord[2] + 0.1

        return -1.0

    def is_waving(self, person):
        keypoints = person.keypoints
        keypoint_threshold = 0.05

        # Draw all the edges
        for edge_pair, edg in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (keypoints[edge_pair[0]].score > keypoint_threshold and
                    keypoints[edge_pair[1]].score > keypoint_threshold):

                dict_limb = {}
                dict_shoulder = {}
                limbs = ["Left wrist", "Left elbow",
                         "Right wrist", "Right elbow"]
                shoulders = ["Left shoulder", "Right shoulder"]

                for pose in keypoints:
                    if body_part_list[pose[0].value] in limbs:
                        coordinate_limb_list = []
                        coordinate_limb_list.append(pose[1].x)
                        coordinate_limb_list.append(pose[1].y)
                        dict_limb[str(body_part_list[pose[0].value])
                                  ] = coordinate_limb_list

                    if body_part_list[pose[0].value] in shoulders:
                        coordinate_shoulder_list = []
                        coordinate_shoulder_list.append(pose[1].x)
                        coordinate_shoulder_list.append(pose[1].y)
                        dict_shoulder[body_part_list[pose[0].value]
                                      ] = coordinate_shoulder_list

                for limb, limb_coordinate_list in dict_limb.items():
                    for shoulder, shoulder_coordinate_list in dict_shoulder.items():
                        if limb == 'Left wrist' and shoulder == 'Left shoulder':
                            # print("left limb y:", limb_coordinate_list[1])
                            # print("left shoulder y:", shoulder_coordiante_list[1])
                            if limb_coordinate_list[1] < shoulder_coordinate_list[1]:
                                print("     -->  Found one waving hand")

                                return True

                        if limb == 'Right wrist' and shoulder == 'Right shoulder':
                            # print("right limb y:", limb_coordinate_list[1])
                            # print("right shoulder y:", shoulder_coordiante_list[1])
                            if limb_coordinate_list[1] < shoulder_coordinate_list[1]:
                                print("     -->  Found one waving hand")

                                return True
        return False

        # ----------------------------------------------------------------------

    def is_pointing(self, person):
        keypoints = person.keypoints
        keypoint_threshold = 0.05

        bPointing = 0
        bRight = 0
        bTop = 0

        thresoldDistance = 2
        # Draw all the edges
        for edge_pair, edg in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (keypoints[edge_pair[0]].score > keypoint_threshold and
                    keypoints[edge_pair[1]].score > keypoint_threshold):

                dict_wrist = {}
                dict_elbow = {}
                dict_shoulder = {}
                dict_hips = {}
                dict_knees = {}
                dict_ankle = {}
                wrists = ["Left wrist", "Right wrist"]
                elbows = ["Left elbow", "Right elbow", "Right elbow"]
                shoulders = ["Left shoulder", "Right shoulder"]
                hips = ['Left hip', 'Right hip']
                knees = ['Left knee', 'Right knee']
                ankles = ['Left ankle', 'Right ankle']

                for pose in keypoints:
                    if body_part_list[pose[0].value] in hips:
                        coordinate_hips_list = []
                        coordinate_hips_list.append(pose[1].x)
                        coordinate_hips_list.append(pose[1].y)
                        dict_hips[str(body_part_list[pose[0].value])
                                  ] = coordinate_hips_list

                    if body_part_list[pose[0].value] in knees:
                        coordinate_knees_list = []
                        coordinate_knees_list.append(pose[1].x)
                        coordinate_knees_list.append(pose[1].y)
                        dict_knees[body_part_list[pose[0].value]
                                   ] = coordinate_knees_list

                    if body_part_list[pose[0].value] in ankles:
                        coordinate_ankles_list = []
                        coordinate_ankles_list.append(pose[1].x)
                        coordinate_ankles_list.append(pose[1].y)
                        dict_ankle[body_part_list[pose[0].value]
                                   ] = coordinate_ankles_list

                    if body_part_list[pose[0].value] in wrists:
                        coordinate_wrist_list = []
                        coordinate_wrist_list.append(pose[1].x)
                        coordinate_wrist_list.append(pose[1].y)
                        dict_wrist[str(body_part_list[pose[0].value])
                                   ] = coordinate_wrist_list

                    if body_part_list[pose[0].value] in elbows:
                        coordinate_elbow_list = []
                        coordinate_elbow_list.append(pose[1].x)
                        coordinate_elbow_list.append(pose[1].y)
                        dict_elbow[str(body_part_list[pose[0].value])
                                   ] = coordinate_elbow_list

                    if body_part_list[pose[0].value] in shoulders:
                        coordinate_shoulder_list = []
                        coordinate_shoulder_list.append(pose[1].x)
                        coordinate_shoulder_list.append(pose[1].y)
                        dict_shoulder[body_part_list[pose[0].value]
                                      ] = coordinate_shoulder_list

                distanceR = 0
                distanceL = 0
                for wrist, coordinate_wrist_list in dict_wrist.items():
                    for shoulder, shoulder_coordinate_list in dict_shoulder.items():
                        for elbow, coordinate_elbow_list in dict_elbow.items():
                            for hip, hips_coordinate_list in dict_hips.items():
                                if(hip == 'Left hip' and wrist == 'Left wrist'):
                                    #print('LEFT :')
                                    #print(hips_coordinate_list)
                                    #print(coordinate_wrist_list)
                                    distanceL = self.getDistance(hips_coordinate_list, coordinate_wrist_list)
                                    print("distance Hip/Wrist LEFT : " +str(distanceL))
                                if (hip == 'Right hip' and wrist == 'Right wrist'):
                                    #print('RIGHT :')
                                    #print(hips_coordinate_list)
                                    #print(coordinate_wrist_list)
                                    distanceR = self.getDistance(hips_coordinate_list, coordinate_wrist_list)
                                    print("distance Hip/Wrist LEFT : " +str(distanceR))

                                if(distanceR > thresoldDistance): bPointing = 1
                                if(distanceL > thresoldDistance): bPointing = 1

                                if(distanceR > distanceL): bRight = 1

                                if wrist == 'Left wrist' and shoulder == 'Left shoulder':
                                    if coordinate_wrist_list[1] < shoulder_coordinate_list[1]: bTop = 1

                                if wrist == 'Right wrist' and shoulder == 'Right shoulder':
                                    if coordinate_wrist_list[1] < shoulder_coordinate_list[1]: bTop = 1

                return bPointing, bRight, bTop

        return bPointing, bRight, bTop
   # ----------------------------------------------------------------------

    def is_sit_down(self, person):
        keypoints = person.keypoints
        keypoint_threshold = 0.05

        # first is to get the keypoint cooridnate y and x
        keypoint_list = []
        for i in range(len(keypoints)):
            keypoint_list.append(keypoints[i].coordinate.y)
            keypoint_list.append(keypoints[i].coordinate.x)
        input_data = np.array(keypoint_list).reshape(1, 34).astype(np.float32)

        self._interpreter_sitdown.allocate_tensors()

        # Run inference with the standing&sitting model.
        self._interpreter_sitdown.set_tensor(
            self._input_details[0]['index'], input_data)
        self._interpreter_sitdown.invoke()

        # Get the model output
        model_output = self._interpreter_sitdown.get_tensor(
            self._output_details[0]['index'])

        # if model_output>0.5:
        #     return False
        # else:
        #     return True

        # Draw all the edges
        for edge_pair, edg in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (keypoints[edge_pair[0]].score > keypoint_threshold and
                    keypoints[edge_pair[1]].score > keypoint_threshold):

                dict_hips = {}
                dict_knees = {}
                dict_ankle = {}
                hips = ['Left hip', 'Right hip']
                knees = ['Left knee', 'Right knee']
                ankles = ['Left ankle', 'Right ankle']

                for pose in keypoints:
                    if body_part_list[pose[0].value] in hips:
                        coordinate_hips_list = []
                        coordinate_hips_list.append(pose[1].x)
                        coordinate_hips_list.append(pose[1].y)
                        dict_hips[str(body_part_list[pose[0].value])
                                  ] = coordinate_hips_list

                    if body_part_list[pose[0].value] in knees:
                        coordinate_knees_list = []
                        coordinate_knees_list.append(pose[1].x)
                        coordinate_knees_list.append(pose[1].y)
                        dict_knees[body_part_list[pose[0].value]
                                   ] = coordinate_knees_list

                    if body_part_list[pose[0].value] in ankles:
                        coordinate_ankles_list = []
                        coordinate_ankles_list.append(pose[1].x)
                        coordinate_ankles_list.append(pose[1].y)
                        dict_ankle[body_part_list[pose[0].value]
                                   ] = coordinate_ankles_list

                for hip, hips_coordinate_list in dict_hips.items():

                    if self.check_outside_frame(hips_coordinate_list):
                        return False
                    for knee, knees_coordinate_list in dict_knees.items():
                        if self.check_outside_frame(knees_coordinate_list):
                            return False
                        for ankle, ankle_coordomate_list in dict_ankle.items():
                            if self.check_outside_frame(ankle_coordomate_list):
                                return False

                            if hip == 'Left hip' and knees == 'Left knee' and ankle == 'Left ankle':
                                angle_knee = self.getAngle((ankle_coordomate_list[0], ankle_coordomate_list[1]), (
                                    knees_coordinate_list[0], knees_coordinate_list[1]), (hips_coordinate_list[0], hips_coordinate_list[1]))
                                # print("Angle knees: ", angle_knee)

                                # ((angle_hip < 190 and angle_hip > 60) or (angle_hip > 190 and angle_hip < 315)) and
                                if (model_output < 0.5):
                                    print("        -->  Person is sitting down")
                                    return True
                                # ((angle_knee < 150 and angle_knee > 50) or (angle_knee > 225 and angle_knee < 315)) and
                                elif (model_output > 0.5):
                                    if (angle_knee < 150 and angle_knee > 50) or (angle_knee > 220 and angle_knee < 315):
                                        return True
                                    else:
                                        return False

                            if hip == 'Right hip' and knee == 'Right knee' and ankle == 'Right ankle':
                                angle_knee = self.getAngle((ankle_coordomate_list[0], ankle_coordomate_list[1]), (
                                    knees_coordinate_list[0], knees_coordinate_list[1]), (hips_coordinate_list[0], hips_coordinate_list[1]))
                                # print("Angle knees: ", angle_knee)

                                # ((angle_hip < 135 and angle_hip > 45) or (angle_hip > 225 and angle_hip < 315)) and
                                if (model_output < 0.5):
                                    print("        -->  Person is sitting down")
                                    return True
                                # ((angle_knee < 170 and angle_knee > 50) or (angle_knee > 225 and angle_knee < 315)) and
                                elif (model_output > 0.5):
                                    if (angle_knee < 150 and angle_knee > 50) or (angle_knee > 220 and angle_knee < 315):
                                        return True
                                    else:
                                        return False
            return False

    # def get_landmark(self, label):
    #     for edge_pair, edg in KEYPOINT_EDGE_INDS_TO_COLOR.items():
    #        shoulders = ['Left shoulder', 'Right shoulder']
    #        hips = ['Left hip', 'Right hip' ]
    #        knees = ['Left knee', 'Right knee']

    #        for pose in keypoints:
    #            if body_part_list[pose[0].value] in hips:
    #                 return

    def check_outside_frame(self, coord):
        if coord[0] > self.camera_dim[0] or coord[1] > self.camera_dim[1]:
            return True
        else:
            return False

    def getAngle(self, a, b, c):
        ang = math.degrees(math.atan2(
            c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang + 360 if ang < 0 else ang

    def getDistance(self, a, b):
        dist =  math.dist(a,b) # math.sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2 )
        return dist #abs(dis)

    def transform_point(self, transformation, point_wrt_source):
        point_wrt_target = tf2_geometry_msgs.do_transform_point(
            PointStamped(point=point_wrt_source), transformation).point
        return [point_wrt_target.x, point_wrt_target.y, point_wrt_target.z]

    def get_transformation(self, source_frame, target_frame):
        # get the tf at first available time
        try:
            transformation = self.tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.5))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr('Unable to find the transformation from %s to %s'
                         % source_frame, target_frame)
        return transformation

    def compute_absolute_pose(self, pose):
        transform = self.get_transformation("CameraTop_optical_frame", "odom")

        p = Point32()
        p.x = pose[0]
        p.y = pose[1]
        p.z = pose[2]

        new_pose = self.transform_point(transform, p)

        return new_pose

    def cleanup(self):
        rospy.loginfo(
            ON_PURPLE+"[RoboBreizh - Tracker] Shutting down vision node"+W)
        self.video_service.unsubscribe(self.videosClient)
        self.session.close()

    # def calculateAngle(self, landmark1, landmark2, landmark3):
    #     '''
    #     This function calculates angle between three different landmarks.
    #     Args:
    #         landmark1: The first landmark containing the x,y and z coordinates.
    #         landmark2: The second landmark containing the x,y and z coordinates.
    #         landmark3: The third landmark containing the x,y and z coordinates.
    #     Returns:
    #         angle: The calculated angle between the three landmarks.

    #     '''

    #     # Get the required landmarks coordinates.
    #     x1, y1, _ = landmark1
    #     x2, y2, _ = landmark2
    #     x3, y3, _ = landmark3

    #     # Calculate the angle between the three points
    #     angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    #     # Check if the angle is less than zero.
    #     if angle &lt; 0:

    #         # Add 360 to the found angle.
    #         angle += 360

    #     # Return the calculated angle.
    #     return angle

    # def classifyPose(self, landmarks, output_image, display=False):

    #     # Initialize the label of the pose. It is not known at this stage.
    #     label = 'Unknown Pose'

    #     # Specify the color (Red) with which the label will be written on the image.
    #     color = (0, 0, 255)

    #     # Calculate the required angles.
    #     #----------------------------------------------------------------------------------------------------------------

    #     # Get the angle between the left shoulder, elbow and wrist points.
    #     left_elbow_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    #                                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
    #                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    #     # Get the angle between the right shoulder, elbow and wrist points.
    #     right_elbow_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    #                                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
    #                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    #     # Get the angle between the left elbow, shoulder and hip points.
    #     left_shoulder_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
    #                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
    #                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    #     # Get the angle between the right hip, shoulder and elbow points.
    #     right_shoulder_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    #                                         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
    #                                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    #     # Get the angle between the left hip, knee and ankle points.
    #     left_knee_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
    #                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
    #                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    #     # Get the angle between the right hip, knee and ankle points
    #     right_knee_angle = self.calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
    #                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
    #                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    #     #----------------------------------------------------------------------------------------------------------------

    #     # Check if it is the warrior II pose or the T pose.
    #     # As for both of them, both arms should be straight and shoulders should be at the specific angle.
    #     #----------------------------------------------------------------------------------------------------------------

    #     # Check if the both arms are straight.
    #     if left_elbow_angle &gt; 165 and left_elbow_angle &lt; 195 and right_elbow_angle &gt; 165 and right_elbow_angle &lt; 195:

    #         # Check if shoulders are at the required angle.
    #         if left_shoulder_angle &gt; 80 and left_shoulder_angle &lt; 110 and right_shoulder_angle &gt; 80 and right_shoulder_angle &lt; 110:

    #     # Check if it is the warrior II pose.
    #     #----------------------------------------------------------------------------------------------------------------

    #             # Check if one leg is straight.
    #             if left_knee_angle &gt; 165 and left_knee_angle &lt; 195 or right_knee_angle &gt; 165 and right_knee_angle &lt; 195:

    #                 # Check if the other leg is bended at the required angle.
    #                 if left_knee_angle &gt; 90 and left_knee_angle &lt; 120 or right_knee_angle &gt; 90 and right_knee_angle &lt; 120:

    #                     # Specify the label of the pose that is Warrior II pose.
    #                     label = 'Warrior II Pose'

    #     #----------------------------------------------------------------------------------------------------------------

    #     # Check if it is the T pose.
    #     #----------------------------------------------------------------------------------------------------------------

    #             # Check if both legs are straight
    #             if left_knee_angle &gt; 160 and left_knee_angle &lt; 195 and right_knee_angle &gt; 160 and right_knee_angle &lt; 195:

    #                 # Specify the label of the pose that is tree pose.
    #                 label = 'T Pose'

    #     #----------------------------------------------------------------------------------------------------------------

    #     # Check if it is the tree pose.
    #     #----------------------------------------------------------------------------------------------------------------

    #     # Check if one leg is straight
    #     if left_knee_angle &gt; 165 and left_knee_angle &lt; 195 or right_knee_angle &gt; 165 and right_knee_angle &lt; 195:

    #         # Check if the other leg is bended at the required angle.
    #         if left_knee_angle &gt; 315 and left_knee_angle &lt; 335 or right_knee_angle &gt; 25 and right_knee_angle &lt; 45:

    #             # Specify the label of the pose that is tree pose.
    #             label = 'Tree Pose'

    #     #----------------------------------------------------------------------------------------------------------------

    #     # Check if the pose is classified successfully
    #     if label != 'Unknown Pose':

    #         # Update the color (to green) with which the label will be written on the image.
    #         color = (0, 255, 0)

    #     # Write the label on the output image.
    #     cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    #     # Return the output image and the classified label.
    #     return output_image, label


if __name__ == '__main__':
    session = qi.Session()
    try:
        session.connect("tcp://127.0.0.1:9559")
    except RuntimeError:
        print("Can't connect to Naoqi")
        sys.exit(1)

    Pose_Detector(session, standalone=True)
