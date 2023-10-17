# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

import math
from typing import List, Tuple
import tf2_geometry_msgs
from geometry_msgs.msg import Point32, PointStamped
import tf2_ros
import cv2
from perception_pepper.models.PoseDetection.roboBreizh_Data_pose import PersonPose
import rclpy
from rclpy.duration import Duration

import numpy as np

# map edges to a RGB color
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

# A list of distictive colors
COLOR_LIST = [
    (47, 79, 79),
    (139, 69, 19),
    (0, 128, 0),
    (0, 0, 139),
    (255, 0, 0),
    (255, 215, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (30, 144, 255),
    (255, 228, 181),
    (255, 105, 180),
]

W = '\033[0m'  # white (normal)
R = '\033[31m'  # red
G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue
P = '\033[35m'  # purple
CYAN = '\033[96m'

camera_dim = [640, 480]

body_part_list = ['Nose', 'Left eye', 'Right eye', 'Left ear', 'Right ear', 'Left shoulder', 'Right shoulder', 'Left elbow',
                  'Right elbow', 'Left wrist', 'Right wrist', 'Left hip', 'Right hip', 'Left knee', 'Right knee', 'Left ankle', 'Right ankle']

def get_height(tf_buffer, person, depth_image):
    keypoints = person.keypoints
    keypoint_threshold = 0.05

    head = ['Left eye', 'Right eye', 'Left ear', 'Right ear']
    for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if (keypoints[edge_pair[0]].score > keypoint_threshold and
                keypoints[edge_pair[1]].score > keypoint_threshold):
            for pose in keypoints:
                if body_part_list[pose[0].value] in head:
                    if check_outside_frame(pose[1]):
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
                        coord = compute_absolute_pose(tf_buffer, 
                            [point_x, point_y, point_z])
                        # print("coord gaze frame odom? ", [coord[0], coord[1], coord[2]])

                        # coord eyes or ears + 10cm for global height
                        return coord[2] + 0.1

    return -1.0

def transform_point(transformation, point_wrt_source):
    point_wrt_target = tf2_geometry_msgs.do_transform_point(
        PointStamped(point=point_wrt_source), transformation).point
    return [point_wrt_target.x, point_wrt_target.y, point_wrt_target.z]

def get_transformation(tf_buffer , source_frame, target_frame):
    rclpy.init()
    duration = Duration(seconds=0.5)

    # get the tf at first available time
    try:
        transformation = tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time() , duration)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logerr('Unable to find the transformation from %s to %s'
                        % source_frame, target_frame)
    rclpy.shutdown()
    return transformation

def compute_absolute_pose(tf_buffer, pose):
  
    transform = get_transformation(tf_buffer, "CameraTop_optical_frame", "odom")
    p = Point32()
    p.x = pose[0]
    p.y = pose[1]
    p.z = pose[2]
    new_pose = transform_point(transform, p)

    return new_pose

def check_outside_frame(coord):
    if coord[0] > camera_dim[0] or coord[1] > camera_dim[1]:
        return True
    else:
        return False

def visualize(
    image: np.ndarray,
    list_persons: List[PersonPose],
    keypoint_color: Tuple[int, ...] = None,
    keypoint_threshold: float = 0.05,
    instance_threshold: float = 0.1,
) -> np.ndarray:
  """Draws landmarks and edges on the input image and return it.

  Args:
    image: The input RGB image.
    list_persons: The list of all "Person" entities to be visualize.
    keypoint_color: the colors in which the landmarks should be plotted.
    keypoint_threshold: minimum confidence score for a keypoint to be drawn.
    instance_threshold: minimum confidence score for a person to be drawn.

  Returns:
    Image with keypoints and edges.
  """
  print("                Nb person in the camera: ", len(list_persons))
  dict_track_wave_person = {}

  for person in list_persons:
    
    if person.score < instance_threshold:
      continue

    keypoints = person.keypoints
    bounding_box = person.bounding_box

    # Assign a color to visualize keypoints.
    if keypoint_color is None:
      if person.id is None:
        # If there's no person id, which means no tracker is enabled, use
        # a default color.
        person_color = (0, 255, 0)
      else:
        # If there's a person id, use different color for each person.
        person_color = COLOR_LIST[person.id % len(COLOR_LIST)]
    else:
      person_color = keypoint_color

    # Draw all the landmarks
    for i in range(len(keypoints)):
      if keypoints[i].score >= keypoint_threshold:
        cv2.circle(image, keypoints[i].coordinate, 2, person_color, 4)

    # Draw all the edges
    for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (keypoints[edge_pair[0]].score > keypoint_threshold and
          keypoints[edge_pair[1]].score > keypoint_threshold):

        start_x = bounding_box.start_point.x
        start_y = bounding_box.start_point.y
        end_x = bounding_box.end_point.x
        end_y = bounding_box.end_point.y
        midpoint_x = int((end_x + start_x) / 2)
        midpoint_y = int((end_y + start_y) / 2)

        dict_track_wave_person[person.id] = (midpoint_x, midpoint_y)

        cv2.circle(image, (midpoint_x,midpoint_y), 10, (0, 0, 255), 2)
        
        cv2.line(image, keypoints[edge_pair[0]].coordinate,
                 keypoints[edge_pair[1]].coordinate, edge_color, 2)

    # Draw bounding_box with multipose
    if bounding_box is not None:
      start_point = bounding_box.start_point
      end_point = bounding_box.end_point
      cv2.rectangle(image, start_point, end_point, person_color, 2)
      # Draw id text when tracker is enabled for MoveNet MultiPose model.
      # (id = None when using single pose model or when tracker is None)
      if person.id:
        id_text = str('id = ' + str(person.id))
        cv2.putText(image, id_text, start_point, cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 2)

  return image

def is_waving(person):
  
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


def is_waving_gpsr(person):
  
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
                        
                          return "waving_left"

                  if limb == 'Right wrist' and shoulder == 'Right shoulder':
                      # print("right limb y:", limb_coordinate_list[1])
                      # print("right shoulder y:", shoulder_coordiante_list[1])
                      if limb_coordinate_list[1] < shoulder_coordinate_list[1]:
                          print("     -->  Found one waving hand")

                          return "waving_right"
  return "No_waving"

def keep_aspect_ratio_resizer(
    image: np.ndarray, target_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
  """Resizes the image.

  The function resizes the image such that its longer side matches the required
  target_size while keeping the image aspect ratio. Note that the resizes image
  is padded such that both height and width are a multiple of 32, which is
  required by the model. See
  https://tfhub.dev/google/tfjs-model/movenet/multipose/lightning/1 for more
  detail.

  Args:
    image: The input RGB image as a numpy array of shape [height, width, 3].
    target_size: Desired size that the image should be resize to.

  Returns:
    image: The resized image.
    (target_height, target_width): The actual image size after resize.

  """
  height, width, _ = image.shape
  if height > width:
    scale = float(target_size / height)
    target_height = target_size
    scaled_width = math.ceil(width * scale)
    image = cv2.resize(image, (scaled_width, target_height))
    target_width = int(math.ceil(scaled_width / 32) * 32)
  else:
    scale = float(target_size / width)
    target_width = target_size
    scaled_height = math.ceil(height * scale)
    image = cv2.resize(image, (target_width, scaled_height))
    target_height = int(math.ceil(scaled_height / 32) * 32)

  padding_top, padding_left = 0, 0
  padding_bottom = target_height - image.shape[0]
  padding_right = target_width - image.shape[1]
  # add padding to image
  image = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left,
                             padding_right, cv2.BORDER_CONSTANT)
  return image, (target_height, target_width)

def getDistance(a, b):
    dist =  math.dist(a,b) # math.sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2 )
    return dist #abs(dis)

def is_pointing(person):
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
                                distanceL = getDistance(hips_coordinate_list, coordinate_wrist_list)
                            if (hip == 'Right hip' and wrist == 'Right wrist'):
                                #print('RIGHT :')
                                #print(hips_coordinate_list)
                                #print(coordinate_wrist_list)
                                distanceR = getDistance(hips_coordinate_list, coordinate_wrist_list)

                            if(distanceR > thresoldDistance): bPointing = 1
                            if(distanceL > thresoldDistance): bPointing = 1

                            if(distanceR > distanceL): bRight = 1

                            if wrist == 'Left wrist' and shoulder == 'Left shoulder':
                                if coordinate_wrist_list[1] < shoulder_coordinate_list[1]: bTop = 1

                            if wrist == 'Right wrist' and shoulder == 'Right shoulder':
                                if coordinate_wrist_list[1] < shoulder_coordinate_list[1]: bTop = 1

            return bPointing, bRight, bTop

    return bPointing, bRight, bTop

#def is_waving(person):

#   keypoints = person.keypoints
#   keypoint_threshold = 0.05

#   # Draw all the edges
#   for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
#     if (keypoints[edge_pair[0]].score > keypoint_threshold and
#         keypoints[edge_pair[1]].score > keypoint_threshold):

#       dict_limb = {}
#       dict_shoulder = {}
#       limbs = ["Left wrist", "Left elbow", "Right wrist", "Right elbow"]
#       shoulders = ["Left shoulder", "Right shoulder"]
#       for pose in keypoints:
#         if body_part_list[pose[0].value] in limbs:
#           coordinate_limb_list = []
#           coordinate_limb_list.append(pose[1].x)
#           coordinate_limb_list.append(pose[1].y)
#           dict_limb[str(body_part_list[pose[0].value])] = coordinate_limb_list
#         if body_part_list[pose[0].value] in shoulders:
#           coordinate_shoulder_list = []
#           coordinate_shoulder_list.append(pose[1].x)
#           coordinate_shoulder_list.append(pose[1].y)
#           dict_shoulder[body_part_list[pose[0].value]] = coordinate_shoulder_list

#       # print(dict_limb)
#       # print(dict_shoulder)
#       # print("-----------------------------------------")
#       # print(person.id)

#       for limb, limb_coordinate_list in dict_limb.items():
#         for shoulder, shoulder_coordiante_list in dict_shoulder.items():
#           if limb == 'Left wrist' and shoulder == 'Left shoulder':
#             # print("left limb y:", limb_coordinate_list[1])
#             # print("left shoulder y:", shoulder_coordiante_list[1])
#             if limb_coordinate_list[1] < shoulder_coordiante_list[1]:
#               print("Found one waving hand")

#               return True

#           if limb == 'Right wrist' and shoulder == 'Right shoulder':
#             # print("right limb y:", limb_coordinate_list[1])
#             # print("right shoulder y:", shoulder_coordiante_list[1])
#             if limb_coordinate_list[1] < shoulder_coordiante_list[1]:
#               print("Found one waving hand")

#               return True
#     return False