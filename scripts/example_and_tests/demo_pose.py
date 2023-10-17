
# create ROS2 node PoseDemo

from .PoseDetection.MoveNet_MultiPose.movenet_multipose import MoveNetMultiPose

import Camera.Naoqi_camera as nc
from Camera.naoqi_camera_types import CameraID, CameraResolution2D as res2D, CameraResolution3D as res3D, ColorSpace2D as cs2D, ColorSpace3D as cs3D

class PoseDemo(Node):
    def __init__(self):
        super().__init__('PoseDemo')
        self.pose_model = MoveNetMultiPose(pose_model_name="movenet_multipose")
        depth_camera_res = res3D.R320x240
        rgb_camera_res = res2D.R320x240
    
        cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
        