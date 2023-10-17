
# create ROS2 node PoseDemo

from perception_pepper.models.PoseDetection import MoveNetMultiPose
from perception_pepper.models.PoseDetection import visualize

from perception_pepper.Camera import NaoqiSingleCamera
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import argparse
class PoseDemo(Node):
    def __init__(self, qi_ip):
        super().__init__('PoseDemo')

        self.ip = qi_ip

        self.pose_model = MoveNetMultiPose(pose_model_name="movenet_multipose")
    
        self.cam = NaoqiSingleCamera(ip=self.ip)
        self.bridge = CvBridge()

        self.pub_cv2 = self.create_publisher(Image, 'pose_detector', 10)
    
    def image_callback(self):
        start = time.time()

        frame = self.cam.get_image('cv2')
        opencv_out = self.inference(frame)

        end = time.time()
        print("FPS: ", 1/(end-start))

        ros_image_yolo_cv = self.bridge.cv2_to_imgmsg(opencv_out, "rgb8")

        self.pub_cv2.publish(ros_image_yolo_cv)

    def inference(self, rgb_image):
        list_person = self.pose_model.inference(rgb_image)
        return visualize(list_person)

def main():
    # get arg 1 and 2
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='demo_indoorVG.onnx', help='model path')
    # parser.add_argument('--res', type=str, default='320', help='resolution')
    # parser.add_argument('--classes', type=str, default='classes.txt', help='classes txt file')
    # parser.add_argument('--ip', type=str, default='127.0.0.1', help='IP robot')


    # args = parser.parse_args()

    # model = args.model
    # res = args.res
    # classes = args.classes
    # ip = args.ip
    rclpy.init()

    # print("Starting detection with args: \n model: ", model, "\n resolution: ", res, "\n")
    PoseDemo('127.0.0.1')
    PoseDemo.image_callback()

if __name__ == '__main__':
    main()
    #rclpy.spin(node)
