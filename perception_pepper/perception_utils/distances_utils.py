#! /usr/bin/env python
# ----------------------------------------------------------------------------
# Authors  : Cedric BUCHE (buche@enib.fr)
# Created Date: 2022
# ---------------------------------------------------------------------------




import rospkg
import math
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Point32, PointStamped
import rospy
import operator
from perception_utils.bcolors import bcolors


######################################################################
# DISTANCE
######################################################################

def detectDistance(depth_image, left, bottom, top, right, camera_info=None):

    # 0,0
    # |          |
    # |          |
    # |          |
    # --------   max X, max Y

    # left,top ------
    # |          |
    # |          |
    # |          |
    # --------right,bottom

    # cut the depth_image a litle bit
    w = int(right - left)
    h = int(bottom - top)
    if w > 0 and h > 0:
        marginh = int(h/4)
        marginw = int(w/4)
    else:
        marginh = 0
        marginw = 0
        print(bcolors.WARNING+"distance w: "+str(w) + " h: "+str(h)+bcolors.ENDC)

    roi_depth = depth_image[int(top+marginh):int(bottom-marginh),
                            int(left+marginw):int(right-marginw)]   # [y:y+h, x:x+w]
    if roi_depth.size == 0:
        print(bcolors.WARNING+'     roi_depth image  EMPTY'+bcolors.ENDC)
        print(bcolors.WARNING+'       depth_image image  starting point ' +
                str(left+marginw) + ',' + str(bottom-marginh)+bcolors.ENDC)
        print(bcolors.WARNING+'       depth_image image  end point ' +
                str(top+marginh) + ','+str(right-marginw)+bcolors.ENDC)
        roi_depth = depth_image

    if roi_depth.shape[0] == 0 or roi_depth.shape[1] == 0:
        print(
            bcolors.WARNING+'    detectDistance roi_depth.shape[0] == 0 or roi_depth.shape[1]'+bcolors.ENDC)
        print(
            bcolors.WARNING+'            + roi_depth.shape[1] : '+str(roi_depth.shape[1])+bcolors.ENDC)
        print(
            bcolors.WARNING+'            + roi_depth.shape[0] : '+str(roi_depth.shape[0])+bcolors.ENDC)
        roi_depth = depth_image

    # compute mean Z
    n = 0
    sum = 0
    for i in range(0, roi_depth.shape[0]):          # height
        for j in range(0, roi_depth.shape[1]):  # width
            value = roi_depth.item(i, j)
            if value > 0.:
                n = n + 1
                sum = sum + value
    if (n != 0):
        mean_z = sum / n
    else:
        mean_z = 0

    mean_distance = mean_z * 0.001  # distance in meters

    # 0,0
    # |          |
    # |          |
    # |          |
    # --------   max X, max Y

    # left,top ------
    # |          |
    # |          |
    # |          |
    # --------right,bottom

    #     Z
    #    /
    #   /
    #  /
    # /
    # ------ > x
    # |
    # |
    # |
    # |
    # y

    #  calculate the middle point
    Xcenter = ((right-left)/2)+left
    Ycenter = ((bottom-top)/2)+top

    # get the distance of that point (Z coordinate)
    Zcenter = mean_distance

    if camera_info == None:
        # ----- 640 x 480 / Cam RGB ----
        # cam_info_msg.K = boost::array<double, 9>{{ 556.845054830986, 0, 309.366895338178, 0, 555.898679730161, 230.592233628776, 0, 0, 1 }};
        m_fx = 556.845054830986
        m_fy = 555.898679730161
        m_cx = 309.366895338178
        m_cy = 230.592233628776

        # ----- 320 x 240 / Cam RGB ----
        #   cam_info_msg.K = boost::array<double, 9>{{ 274.139508945831, 0, 141.184472810944, 0, 275.741846757374, 106.693773654172, 0, 0, 1 }};
        # m_fx = 274.139508945831
        # m_fy = 275.741846757374
        # m_cx = 141.184472810944
        # m_cy = 106.693773654172

        # ----- 640 x 480 / Cam D ----
        #  cam_info_msg.K = boost::array<double, 9>{{ 525, 0, 319.5000000, 0, 525, 239.5000000000000, 0, 0, 1  }};
        # m_fx = 525
        # m_fy = 525
        # m_cx = 319.5000000
        # m_cy = 239.5000000000000

        # ----- 320 x 240 / Cam D ----
        #   cam_info_msg.K = boost::array<double, 9>{{ 525/2.0f, 0, 319.5000000/2.0f, 0, 525/2.0f, 239.5000000000000/2.0f, 0, 0, 1  }};
        # m_fx = 525./2.
        # m_fy = 525./2.
        # m_cx = 319.5000000/2.
        # m_cy = 239.5000000000000/2.

    else:
        m_fx = camera_info.K[0]
        m_fy = camera_info.K[4]
        m_cx = camera_info.K[2]
        m_cy = camera_info.K[5]

    # camera
    #camera_info_K = np.array(camera_info.K)

    # Intrinsic camera matrix for the raw (distorted) images.
    #     [fx  0 cx]
    # K = [ 0 fy cy]
    #     [ 0  0  1]

    inv_fx = 1. / m_fx
    inv_fy = 1. / m_fy

    # pinhole model of a camera
    point_z = Zcenter
    point_x = ((Xcenter-m_cx) * Zcenter) / m_fx
    point_y = ((Ycenter-m_cy) * Zcenter) / m_fy

    dist = math.sqrt(point_x * point_x + point_y *
                     point_y + point_z * point_z)

    return dist, point_x, point_y, point_z, Xcenter, Ycenter


def detectDistanceResolution(depth_image, left, bottom, top, right, resolutionRGB=[320, 240], camera_info=None):

    # 0,0
    # |          |
    # |          |
    # |          |
    # --------   max X, max Y

    # left,top ------
    # |          |
    # |          |
    # |          |
    # --------right,bottom

    # cut the depth_image a litle bit
    #depth_image = depth_image[240, 320]
    resolutionD_height = depth_image.shape[0]
    resolutionD_width = depth_image.shape[1]
    # resolutionRGB = [640, 480]
    resolutionRGB_height = resolutionRGB[1]
    resolutionRGB_width = resolutionRGB[0]

    ratio_height = resolutionD_height / resolutionRGB_height
    ratio_width = resolutionD_width / resolutionRGB_width
    #print("ratios: ", [ratio_height, ratio_width])
    # index_height = int(ratio_height * resolutionD_height)
    # index_width = int(ratio_width * resolutionD_width)
    #print("indexes: ", [index_height, index_width])

    w = int(right - left)
    h = int(bottom - top)
    if w > 0 and h > 0:
        marginh = int(h/4)
        marginw = int(w/4)
    else:
        marginh = 0
        marginw = 0
        print(bcolors.WARNING+"distance w: "+str(w) + " h: "+str(h)+bcolors.ENDC)

    roi_depth = depth_image[int((top+marginh)*ratio_height):int((bottom-marginh)*ratio_height),
                            int((left+marginw)*ratio_width):int((right-marginw)*ratio_width)]   # [y:y+h, x:x+w]

    #print("roi depth size:", roi_depth.shape)
    if roi_depth.size == 0:
        print(bcolors.WARNING+'     roi_depth image  EMPTY'+bcolors.ENDC)
        print(bcolors.WARNING+'       depth_image image  starting point ' +
                str(left+marginw) + ',' + str(bottom-marginh)+bcolors.ENDC)
        print(bcolors.WARNING+'       depth_image image  end point ' +
                str(top+marginh) + ','+str(right-marginw)+bcolors.ENDC)
        roi_depth = depth_image

    if roi_depth.shape[0] == 0 or roi_depth.shape[1] == 0:
        print(
            bcolors.WARNING+'    detectDistance roi_depth.shape[0] == 0 or roi_depth.shape[1]'+bcolors.ENDC)
        print(
           bcolors.WARNING+'            + roi_depth.shape[1] : '+str(roi_depth.shape[1])+bcolors.ENDC)
        print(
            bcolors.WARNING+'            + roi_depth.shape[0] : '+str(roi_depth.shape[0])+bcolors.ENDC)
        roi_depth = depth_image

    mean_distance = np.matrix.mean(np.matrix(roi_depth))*0.001  # distance in meters

    # 0,0
    # |          |
    # |          |
    # |          |
    # --------   max X, max Y

    # left,top ------
    # |          |
    # |          |
    # |          |
    # --------right,bottom

    #     Z
    #    /
    #   /
    #  /
    # /
    # ------ > x
    # |
    # |
    # |
    # |
    # y

    #  calculate the middle point
    Xcenter = ((right-left)/2)+left
    Ycenter = ((bottom-top)/2)+top

    # get the distance of that point (Z coordinate)
    Zcenter = mean_distance

    if camera_info == None:
        if resolutionRGB_width == 640:
            # ----- 640 x 480 / Cam RGB ----
            # cam_info_msg.K = boost::array<double, 9>{{ 556.845054830986, 0, 309.366895338178, 0, 555.898679730161, 230.592233628776, 0, 0, 1 }};
            m_fx = 556.845054830986
            m_fy = 555.898679730161
            m_cx = 309.366895338178
            m_cy = 230.592233628776
            
        elif resolutionRGB_width == 320:

            # ----- 320 x 240 / Cam RGB ----
            #   cam_info_msg.K = boost::array<double, 9>{{ 274.139508945831, 0, 141.184472810944, 0, 275.741846757374, 106.693773654172, 0, 0, 1 }};
            m_fx = 274.139508945831
            m_fy = 275.741846757374
            m_cx = 141.184472810944
            m_cy = 106.693773654172
        
        else:
            
            rospy.loginfo("Current RGB resolution not supporting distance calculation")

        # ----- 640 x 480 / Cam D ----
        #  cam_info_msg.K = boost::array<double, 9>{{ 525, 0, 319.5000000, 0, 525, 239.5000000000000, 0, 0, 1  }};
        # m_fx = 525
        # m_fy = 525
        # m_cx = 319.5000000
        # m_cy = 239.5000000000000

        # ----- 320 x 240 / Cam D ----
        #   cam_info_msg.K = boost::array<double, 9>{{ 525/2.0f, 0, 319.5000000/2.0f, 0, 525/2.0f, 239.5000000000000/2.0f, 0, 0, 1  }};
            m_fx = 525./2.
            m_fy = 525./2.
            m_cx = 319.5000000/2.
            m_cy = 239.5000000000000/2.

    else:
        m_fx = camera_info.K[0]
        m_fy = camera_info.K[4]
        m_cx = camera_info.K[2]
        m_cy = camera_info.K[5]

    # camera
    #camera_info_K = np.array(camera_info.K)

    # Intrinsic camera matrix for the raw (distorted) images.
    #     [fx  0 cx]
    # K = [ 0 fy cy]
    #     [ 0  0  1]


    # pinhole model of a camera
    point_z = Zcenter
    point_x = ((Xcenter-m_cx) * point_z) / m_fx
    point_y = ((Ycenter-m_cy) * point_z) / m_fy

    dist = np.linalg.norm(np.array([point_x,point_y,point_z]))

    return dist, point_x, point_y, point_z, Xcenter, Ycenter