#! /usr/bin/env python
# ----------------------------------------------------------------------------
# Authors  : Cedric BUCHE (buche@enib.fr)
# Created Date: 2022
# ---------------------------------------------------------------------------

import math
import numpy as np
import tf2_ros
import tf2_geometry_msgs
import rospy
from geometry_msgs.msg import Point32, PointStamped


######################################################################
# Transform
######################################################################


# Return the angle between the camera frame and the target in radians
def computeAngle(image_width, x, w):
    Xcenter = (w/2)+x
    person_position = Xcenter-(image_width/2)
    HFOV = 57.2
    theta = HFOV*person_position/image_width

    return math.radians(theta)


def compute_yaw(x):
    HFOV = math.radians(55.2)
    dist = x-320/2
    return dist*HFOV/320


def compute_pitch(y):
    VFOV = math.radians(44.3)
    dist = y-240/2
    return dist*VFOV/240


def compute_absolute_pose(pose) -> Point32:
    transform = get_transformation("CameraTop_optical_frame", "odom")

    p = Point32()
    p.x = pose[0]
    p.y = pose[1]
    p.z = pose[2]

    new_pose = transform_point(transform, p)

    return new_pose

def transform_point(transformation, point_wrt_source:Point32) -> Point32:
    point_wrt_target = \
        tf2_geometry_msgs.do_transform_point(PointStamped(point=point_wrt_source),
                                             transformation).point
    p = Point32()
    p.x = point_wrt_target.x
    p.y = point_wrt_target.y
    p.z = point_wrt_target.z
    return p

def get_transformation(source_frame, target_frame,tf_cache_duration=7.0):
    # get the tf at first available time
    try:
        tf_buffer = tf2_ros.Buffer(rospy.Duration(tf_cache_duration))
        tf2_ros.TransformListener(tf_buffer)
        transformation = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(tf_cache_duration))
        return transformation

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException):
        rospy.logerr(f'Unable to find the transformation from {source_frame} to {target_frame}')

def point_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z])


def point_to_numpy_2D(msg):
    return np.array([msg.x, msg.y])

