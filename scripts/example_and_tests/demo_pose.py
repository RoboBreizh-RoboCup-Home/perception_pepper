
# create ROS2 node PoseDemo

from .PoseDetection.MoveNet_MultiPose.movenet_multipose import MoveNetMultiPose

import Camera.Naoqi_camera as nc
from Camera.naoqi_camera_types import CameraID, CameraResolution2D as res2D, CameraResolution3D as res3D, ColorSpace2D as cs2D, ColorSpace3D as cs3D

import argparse
class PoseDemo(Node):
    def __init__(self):
        super().__init__('PoseDemo')
        self.pose_model = MoveNetMultiPose(pose_model_name="movenet_multipose")
        depth_camera_res = res3D.R320x240
        rgb_camera_res = res2D.R320x240
    
        cameras = nc.NaoqiCameras(ip=qi_ip, resolution = [rgb_camera_res, depth_camera_res])
        
def main():
    rclpy.init()
    node = PoseDemo()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    # get arg 1 and 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='demo_indoorVG.onnx', help='model path')
    parser.add_argument('--res', type=str, default='320', help='resolution')
    parser.add_argument('--classes', type=str, default='classes.txt', help='classes txt file')
    parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP robot')


    args = parser.parse_args()
    model = args.model
    res = args.res
    classes = args.classes
    ip = args.ip

    print("Starting detection with args: \n model: ", model, "\n resolution: ", res, "\n")
    Detector(model, res, classes, ip)
    Detector.image_callback()
    #rclpy.spin(node)
