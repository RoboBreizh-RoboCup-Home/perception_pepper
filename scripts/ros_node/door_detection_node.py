#! /usr/bin/env python
# ----------------------------------------------------------------------------
# Authors  : Cedric BUCHE (buche@enib.fr)
# Created Date: 2022
# ---------------------------------------------------------------------------


from enum import Enum
import sys
import rospy
import actionlib

# MSG
from sensor_msgs.msg import Range
from std_msgs.msg import Float32

# Robobreizh
from perception_utils.bcolors import bcolors
from robobreizh_msgs.msg import SonarAction, SonarFeedback, SonarResult

class DoorState():
    OPEN = 0
    CLOSE = 1

class MinFrontValueDetector():
    # messages that are published feedback and result
    _door_detection_feedback = SonarFeedback()
    _door_detection_result = SonarResult()
    # _door_state : DoorState()
    _action_name = "/robobreizh/door_detection_action"
    _sonar_distance = 0

    def __init__(self, min_distance_door):
        self.min_distance = min_distance_door

        # Action server that would be call to start and stop the detection
        self._as = actionlib.SimpleActionServer(
            self._action_name, SonarAction, execute_cb=self.door_detection, auto_start=False)

        # starts the action server
        self._as.start()

    def door_detection(self, goal):
        """
        Action server for the door detection
        """
        # Subscriber to the sonar information
        self._sonar_listener = rospy.Subscriber(
            '/naoqi_driver/sonar/front', Range, self.sonarCallback, queue_size=1)
        r = rospy.Rate(1)
        success = True

        # Run code until it timesout or is canceled
        timeout = rospy.Duration.from_sec(goal.timeout)
        beginning = rospy.get_rostime()
        while rospy.get_rostime() - beginning < timeout:
            # look if there is on obstacle within the set range
            if self._sonar_distance < self.min_distance:
                print(f'{self._sonar_distance} < {self.min_distance}')
                self._door_state = DoorState.CLOSE
                rospy.loginfo(bcolors.BACKRED +
                              "[RoboBreizh - Door] door  " + bcolors.ENDC)
            else:
                self._door_state = DoorState.OPEN
                rospy.loginfo(bcolors.BACKGREEN +
                              "[RoboBreizh - Door] clear " + bcolors.ENDC)

            # Publish feedback
            self._door_detection_feedback.state = self._door_state
            self._as.publish_feedback(self._door_detection_feedback)

            # What to do if the action server receives a cancel message
            if self._as.is_preempt_requested():
                self._sonar_listener.unregister()
                rospy.loginfo(f'{self._action_name}: Preempted')
                self._as.set_preempted()
                success = False
                # force to timeout
                timeout = rospy.Duration.from_sec(0)

            # If the door is open then end the detection
            if self._door_state == DoorState.OPEN:
                timeout = rospy.Duration.from_sec(0)

            # waits 1 second
            r.sleep()

        if success:
            rospy.loginfo(f'{self._action_name}: request succeeded')
            self._door_detection_result = self._door_detection_feedback.state
            self._as.set_succeeded(self._door_detection_result)
        self._sonar_listener.unregister()

    def sonarCallback(self, data):
        """
        Subscriber callback for the sonar information
        """
        self._sonar_distance = data.range


if __name__ == "__main__":

    rospy.init_node('robobreizh_open_door_node')
    MIN_DISTANCE_DOOR = rospy.get_param('~MIN_DISTANCE_DOOR')
    MinFrontValueDetector(MIN_DISTANCE_DOOR)
    rospy.spin()
