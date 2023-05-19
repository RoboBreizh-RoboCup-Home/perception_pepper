import tf2_geometry_msgs
from geometry_msgs.msg import Point32, PointStamped
import tf2_ros
import rospy

camera_dim = [640, 480]

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
    # get the tf at first available time
    try:
        transformation = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.5))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logerr('Unable to find the transformation from %s to %s'
                        % source_frame, target_frame)
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