#! /usr/bin/env python
# ----------------------------------------------------------------------------
# Authors  : Cedric BUCHE (buche@enib.fr)
# Created Date: 2022
# ---------------------------------------------------------------------------
   
# OPENCV
import cv2
import numpy as np

import rospy
from visualization_msgs.msg import Marker, MarkerArray
import perception_utils.transform_utils as transform_utils 
from robobreizh_msgs.msg import ObjectList, Object
from geometry_msgs.msg import Point32

######################################################################
global        font
global        fontScale
global        thickness
global        colorWhite
   

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = .3
thickness = 1

colorWhite = [255,255,255]
   
######################################################################
# DISPLAY
######################################################################

def createImageKMEANS(cropped,color,mappingImages,mapping,color_str,classe):
        mapping2 = cv2.resize(mapping, (cropped.shape[1], cropped.shape[0]), interpolation=cv2.INTER_AREA)
        cv2.rectangle(mapping2, (0, 0), (cropped.shape[1], 15), color, thickness=-1)
        cv2.putText(mapping2, str(classe), (10, 10), font,fontScale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.putText(mapping2, color_str, (int(10), int(cropped.shape[0] / 2)), font, fontScale, color, thickness, cv2.LINE_AA)
        mappingImages.append(mapping2)

        return mappingImages

def manageDisplay(score, left, top, right, bottom, cv_rgb, cv_depth, dist, classe, point_x, point_y, point_z, color):
        # left,top ------
        # |          |
        # |          |
        # |          |
        # --------right,bottom

        text_classe_confidence = "{}: {:.2f}%".format(classe, score*100)

        # rectangle
        # (startX, startY), (endX, endY)
        cv2.rectangle(cv_rgb, (left, top), (right, bottom),color, thickness)

        # text classe / confidence
        y_text = top - 5 if top - 5 > 5 else top + 5
        max_x = left + 150 if left + 150 < right else right - 5

        cv2.rectangle(cv_rgb, (int(left), int(y_text-10)), (int(max_x),int(top)), color, thickness=-1)		# background
        cv2.putText(cv_rgb, text_classe_confidence, (int(left), int(y_text)), font, fontScale, (0, 0, 0), thickness, cv2.LINE_AA)

        # distance

        ratio = 0.5 			# resolutionD vs resolution
        left = int(left * ratio)
        right = int(right * ratio)
        top = int(top * ratio)
        bottom = int(bottom * ratio)

        if cv_depth.any():  # != None :
            cv2.rectangle(cv_depth, (left+10, top+10),(right-10, bottom-10), colorWhite, thickness=1)

            x_str = "X: " + str(format(point_x, '.2f'))
            y_str = "Y: " + str(format(point_y, '.2f'))
            z_str = "Z: " + str(format(point_z, '.2f'))

            cv2.putText(cv_depth, x_str, (left+10, top + 20), font,fontScale, colorWhite, thickness, cv2.LINE_AA)
            cv2.putText(cv_depth, y_str, (left+10, top + 30), font,fontScale, colorWhite, thickness, cv2.LINE_AA)
            cv2.putText(cv_depth, z_str, (left+10, top + 40), font,fontScale, colorWhite, thickness, cv2.LINE_AA)

            dist_str_space = 60
            dist_str = "dist:" + str(format(dist, '.2f')) + "m"
            cv2.putText(cv_depth, dist_str, (left+10, top + dist_str_space), cv2.FONT_HERSHEY_SIMPLEX,fontScale, colorWhite, thickness, cv2.LINE_AA)

def manageDisplayColor(mappingImages):
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
        mapping = cv2.copyMakeBorder(mapping, top2, bottom2, 5, 5, cv2.BORDER_CONSTANT)
        if (i == 0):
            mapping_concat = mapping
        else:
            mapping_concat = np.concatenate( (mapping_concat, mapping), axis=1)
                
    return mapping_concat

def create_obj_marker( obj:Object, id:int, DURATION:int):
    """
    Creates an Rviz Marker associated with an object and converts it from CameraTop_optical_frame to odom 
    """
    # Get Obj Info
    obj_coord:Point32 = obj.coord  # is Point32() on odom frame
    od_to_cam_tf = transform_utils.get_transformation("CameraTop_optical_frame", "base_link")
    cam_pose = transform_utils.transform_point(od_to_cam_tf, obj_coord)
    cam_dist = cam_pose.z  # cam optical frame z
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
    marker.pose.position.x = obj.coord.x
    marker.pose.position.y = obj.coord.y
    marker.pose.position.z = obj.coord.z

    # orientation
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Marker Lifetime
    marker.lifetime = rospy.Duration(DURATION)
    return marker

def create_text_marker(obj:Object, id:int, DURATION:int):
    """
    Create ros rviz text marker associated to an object message
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

def show_RViz_marker_arr(publisher:rospy.Publisher, objects_list:ObjectList, DURATION:int=50):
    """
    Creates a rviz marker for each object in a object list that last a certain duration
    """
    marker_lst = MarkerArray()

    rospy.logdebug(len(objects_list.object_list))
    for i, obj in enumerate(objects_list.object_list):
        # graphical marker id: 0, 2, 4,..
        obj_marker_id = i*2

        # text label id: 1, 3, 5,..
        text_marker_id = 1 + i*2

        # Create text and object markers
        marker = create_obj_marker(obj, obj_marker_id, DURATION)
        text_marker = create_text_marker(obj, text_marker_id, DURATION)

        # added to the marker array
        marker_lst.markers.append(marker)
        marker_lst.markers.append(text_marker)

    publisher.publish(marker_lst)
    return