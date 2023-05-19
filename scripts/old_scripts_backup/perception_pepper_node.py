import rospy

import perception_utils.bcolors

import Camera.Naoqi_camera as Naoqi_camera
import Camera.naoqi_camera_types as nc_types

class PerceptionPepperNode():
    def __init__(self) -> None:
        # create naoqi camera object and instanciate to default variables
        # check the class code and comments if you want more information on initial parameters
        self._cameras = Naoqi_camera.NaoqiCameras()


if __name__ == "__main__":
    """
    Roslaunch entry point for the perception node.
    Here should be called different services as modules to perform specific tasks.
    """
    # declare node
    rospy.init_node("robobreizh_perception_node")

    DESCRIBE_PERSON = rospy.get_param('~DESCRIBE_PERSON')
    DETECT_AGE = rospy.get_param('~DETECT_AGE')
    DETECT_COLOR = rospy.get_param('~DETECT_COLOR')
    DETECT_DISTANCE = rospy.get_param('~DETECT_DISTANCE')
    DISPLAY_DELAY = rospy.get_param('~DISPLAY_DELAY')
    DISPLAY = rospy.get_param('~DISPLAY')

    POSTURE_PERSON = rospy.get_param('~posture')
    WAVE_HAND = rospy.get_param('~wave')

    PerceptionPepperNode()

    # keeps the process running until ctrl-c is pressed or the master shuts it down
    rospy.spin()
    