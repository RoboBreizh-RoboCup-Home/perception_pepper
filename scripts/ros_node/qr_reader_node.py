#!/usr/bin/env python

# import roslib
import rospy
from std_msgs.msg import String
import cv2
import numpy as np
from robobreizh_msgs.msg import *
from robobreizh_msgs.srv import *
import tf2_ros
import time
from perception_utils.bcolors import bcolors
import qi

class QRReader():
        
    def __init__(self, qi_ip) -> None:
        
        self.session = qi.Session()
        self.session.connect("tcp://" + qi_ip)
        
        rospy.loginfo(
            bcolors.CYAN+"[RoboBreizh - Vision]   QR Reader initialisation "+bcolors.ENDC)
        
        self.initQRReaderService()
        
    def initQRReaderService(self):
        rospy.Service('/robobreizh/perception_pepper/qr_reader',
                        qr_to_text, self.handle_ServicePerceptionObject)
            
        rospy.loginfo(
            bcolors.O+"[RoboBreizh - Vision]        Starting Reading QR Code. "+bcolors.ENDC)
        
        rospy.spin()

    def qrReader(self):

        barcode_service = self.session.service("ALBarcodeReader")
        memory_service = self.session.service("ALMemory")

        barcode_service.subscribe("test_barcode")
        
        while True:
            data = memory_service.getData("BarcodeReader/BarcodeDetected")
            if data is None:
                rospy.loginfo("No QR Code is Detected")
            else:
                break
                
            time.sleep(0.3)
        
        return data[0][0]
        
    def handle_ServicePerceptionObject(self, qr_to_text):
                
        result = self.qrReader()
        
        print(result)
                        
        return result
    
if __name__ == "__main__":
    
    rospy.init_node('qr_reader_node', anonymous=True)

    qi_ip = rospy.get_param('~qi_ip')
 
    QRReader(qi_ip)