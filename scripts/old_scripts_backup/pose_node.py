import rospy
from PoseDetection.MoveNet_MultiPose import MoveNetMultiPose
from Camera.naoqi_camera_utils import CameraID, ColorSpace2D, CameraResolution
from Camera.Naoqi_camera import NaoqiSingleCamera

class PoseDetectionNode():
    def __init__(self):
        camera = NaoqiSingleCamera(res=CameraResolution.R640x480, cam=CameraID.TOP, fps=30)
        camera.get_image('cv2')
        model = MoveNetMultiPose()

if "__main__" == __name__:
    rospy.init_node("pose_detection_node")
    rospy.spin()