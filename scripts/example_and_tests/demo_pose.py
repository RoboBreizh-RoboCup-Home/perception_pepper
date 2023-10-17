
# create ROS2 node PoseDemo

from .PoseDetection.MoveNet_MultiPose.movenet_multipose import MoveNetMultiPose


class PoseDemo(Node):
    def __init__(self):
        super().__init__('PoseDemo')
        self.pose_model = MoveNetMultiPose(pose_model_name="movenet_multipose")