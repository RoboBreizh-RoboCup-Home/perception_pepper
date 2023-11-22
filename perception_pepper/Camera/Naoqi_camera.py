import cv2
import numpy as np
from PIL import Image as ImagePIL
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import qi
import rclpy
from rclpy.time import Time

from typing import Union, List

from perception_pepper.Camera.naoqi_camera_types import CameraID, CameraResolution2D as res2D, CameraResolution3D as res3D, ColorSpace2D as cs2D, ColorSpace3D as cs3D


class NaoqiCamera():
    # default cam top RGB at 30 fps on localhost
    def __init__(self, resolution: Union[res2D, res3D] = [res2D.R640x480,res3D.R320x240], camera_id: CameraID = CameraID.TOP, fps: int = 30, color_space: Union[cs2D, cs3D] = cs2D.RGBColorSpace):

        self.session = None
        self.video_service = None
        self.videosClient = None
        

        if camera_id == CameraID.BOTTOM and resolution == res2D.R1280x960:
            raise f'The camera {camera_id} does not support resolution {resolution}'
        if camera_id == CameraID.BOTTOM and resolution == res2D.R2560x1920:
            raise f'The camera {camera_id} does not support resolution {resolution}'

        # make sure resolution and camera id match
        assert isinstance(resolution, res2D) and camera_id != CameraID.DEPTH or isinstance(resolution, res3D) and camera_id != CameraID.DEPTH, f'The resolution {resolution} and camera {camera_id} are not compatible'
        self.camera = camera_id
        # make sure if resolution is type res2D then colorspace is cs2D
        # or is type res3D then colorspace is cs3D
        assert isinstance(resolution, res2D) and isinstance(color_space, cs2D) or isinstance(resolution, res3D) and isinstance(color_space, cs3D), f'The resolution {resolution} and color_space {color_space} are not compatible'

        self.resolution = resolution
        self.color_space = color_space 

        # Settinf up fps
        if self.camera == CameraID.DEPTH:
            if fps > 20:
                self.fps = 20
            else:
                self.fps = fps
        else:
            if fps > 30:
                self.fps = 30
            elif self.resolution == 3 or self.resolution == 4 and fps > 1:
                self.fps = 1
            else:
                self.fps = fps

    def get_image(self, out_format: str):
        pass

    def cleanup(self):
        pass


class NaoqiSingleCamera(NaoqiCamera):
    # default cam top RGB at 30 fps
    def __init__(self, resolution: res2D=res2D.R640x480, camera_id: CameraID = CameraID.TOP, fps: int = 30, color_space: cs2D= cs2D.RGBColorSpace, ip: str = "127.0.0.1"):
        super().__init__(resolution, camera_id, fps, color_space)
        self.session = qi.Session()
        self.session.connect("tcp://"+ip+":9559")

        self.video_service = self.session.service("ALVideoDevice")
        self.videosClient = self.video_service.subscribeCamera("camera_pepper", self.camera.value, self.resolution.value, self.color_space.value, fps)

    def get_image(self, out_format='cv2')->List:
        """
        Input: out_format: 'cv2' or 'PIL'
        Returns a list of images, one for each camera subscribed depending on the out_format.
        """
        raw_rgb_image = self.video_service.getImageRemote(self.videosClient)  # return array of images

        if out_format == 'PIL':
            # Get the image size and pixel array.
            imageWidth = raw_rgb_image[0]
            imageHeight = raw_rgb_image[1]
            array = raw_rgb_image[6]
            image_string = str(bytearray(array))
            # Create a PIL Image from our pixel array.
            rgb_image = ImagePIL.fromstring("RGB", (imageWidth, imageHeight), image_string)
            
        elif out_format == 'cv2':
            rgb_image = np.frombuffer(raw_rgb_image[6], np.uint8).reshape(raw_rgb_image[1], raw_rgb_image[0], 3)
        return rgb_image

    def cleanup(self):
        self.video_service.unsubscribe(self.videosClient)
        self.session.close()


# two cameras for tracking 
class NaoqiCameras():
    def __init__(self, resolution:List[Union[res2D,res3D]]=[res2D.R640x480,res3D.R320x240], camera_id: List[CameraID]=[CameraID.TOP,CameraID.DEPTH], fps:int=20, color_space: List[Union[cs2D,cs3D]]=[cs2D.RGBColorSpace, cs3D.DepthColorSpace],ip: str = "127.0.0.1"):  # default cam top RGB at 30 fps
        assert len(resolution) == len(camera_id) == len(color_space), "All lists must have the same length"
        self.cameras = []

        # test if the input values are correct
        for i in range(len(resolution)):
            self.cameras.append(NaoqiCamera(resolution[i], camera_id[i], fps, color_space[i]))

        self.session = qi.Session()
        self.session.connect(f"tcp://{ip}:9559")
        
        self.bridge = CvBridge()

        self.video_service = self.session.service("ALVideoDevice")

        self.cameras_id = []
        self.cameras_color_space = []
        self.cameras_res = []
        fps = 20  # max fps posssible for 3D camera
        for i in range(len(self.cameras)):
            self.cameras_id.append(self.cameras[i].camera)
            self.cameras_res.append(self.cameras[i].resolution)
            self.cameras_color_space.append(self.cameras[i].color_space)
            # taking the min fps of all cameras for synchro
            if self.cameras[i].fps < fps:
                fps = self.cameras[i].fps
                
        self.videosClient = self.video_service.subscribeCameras( "cameras_pepper", [((self.cameras_id)[0].value), ((self.cameras_id)[1].value)], [self.cameras_res[0].value,self.cameras_res[1].value], [self.cameras_color_space[0].value,self.cameras_color_space[1].value], fps)

    def convertImage3DPepperToCV(self, pepperImageD):
        kDepthColorSpace = 17
        encoding = ""
        img = Image()
        img.header.frame_id = "camera_depth_frame"
        img.height = pepperImageD[1]
        img.width = pepperImageD[0]
        nbLayers = pepperImageD[2]
        if pepperImageD[3] == kDepthColorSpace:
            encoding = "mono16"
        img.encoding = encoding
        img.step = img.width * nbLayers
        img.data = pepperImageD[6]

        return img

    def get_image(self, out_format='cv2')->List:
        """
        Input: out_format: 'cv2' or 'PIL'
        Returns a list of images, one for each camera subscribed depending on the out_format.
        """
        [raw_rgb_images, raw_depth_images] = self.video_service.getImagesRemote(self.videosClient)  # return array of images

        # if out_format == 'PIL':
        #     # Get the image size and pixel array.
        #     imageWidth = raw_rgb_images[0]
        #     imageHeight = raw_rgb_images[1]
        #     array = raw_rgb_images[6]
        #     image_string = str(bytearray(array))
        #     # Create a PIL Image from our pixel array.
        #     rgb_image_pil = ImagePIL.fromstring("RGB", (imageWidth, imageHeight), image_string)
            
        # elif out_format == 'cv2':
        rgb_image_cv2 = np.frombuffer(raw_rgb_images[6], np.uint8).reshape(raw_rgb_images[1], raw_rgb_images[0], 3)
        # depth_image_cv2 = np.frombuffer(raw_depth_images[6], np.uint8).reshape(raw_depth_images[1], raw_depth_images[0], 2)
        
        imgD = self.convertImage3DPepperToCV(raw_depth_images)
        depth_image_cv2 = self.bridge.imgmsg_to_cv2(imgD, "32FC1")
        rgb_image_cv2 = cv2.cvtColor(rgb_image_cv2, cv2.COLOR_BGR2RGB)

        return rgb_image_cv2, depth_image_cv2

    def cleanup(self):
        print("Cleaning up...")
        self.video_service.unsubscribe(self.videosClient)
        self.session.close()

##########################################################################
############################### 2D CAMERA ################################
##########################################################################
# 2D Cameras fps resolution

# Resolution            ID Value    local	Gb Ethernet	100Mb Ethernet	WiFi g
# 40x30     (QQQQVGA)	    8       30fps	        30fps       30fps	30fps
# 80x60     (QQQVGA)	    7       30fps	        30fps	    30fps	30fps
# 160x120   (QQVGA)	        0       30fps	        30fps	    30fps	30fps
# 320x240   (QVGA)	        1       30fps	        30fps	    30fps	11fps
# 640x480   (VGA)	        2       30fps	        30fps	    12fps	2.5fps
# 1280x960  (4VGA)	        3       29fps	        10fps	    3fps	0.5fps

# Pepper camera indexes

# Parameter ID Name	    ID Value	    Type
# AL::kTopCamera	        0	    2D Camera
# AL::kBottomCamera	        1       2D Camera
# AL::kDepthCamera	        2	    Reconstructed 3D Sensor
# AL::kStereoCamera	        3       Stereo Camera


# 2D cameras supported colorspaces

# Parameter ID Name	    ID Value	Number of layers	Number of channels	Description
# AL::kYuvColorSpace	    0	1	1	Buffer only contains the Y (luma component) equivalent to one unsigned char
# AL::kyUvColorSpace	    1	1	1	Buffer only contains the U (Chrominance component) equivalent to one unsigned char
# AL::kyuVColorSpace	    2	1	1	Buffer only contains the V (Chrominance component) equivalent to one unsigned char
# AL::kRgbColorSpace	    3	1	1	Buffer only contains the R (Red component) equivalent to one unsigned char
# AL::krGbColorSpace	    4	1	1	Buffer only contains the G (Green component) equivalent to one unsigned char
# AL::krgBColorSpace	    5	1	1	Buffer only contains the B (Blue component) equivalent to one unsigned char
# AL::kHsyColorSpace	    6	1	1	Buffer only contains the H (Hue component) equivalent to one unsigned char
# AL::khSyColorSpace	    7	1	1	Buffer only contains the S (Saturation component) equivalent to one unsigned char
# AL::khsYColorSpace	    8	1	1	Buffer only contains the Y (Brightness component) equivalent to one unsigned char
# AL::kYUV422ColorSpace	    9	2	2	Native format, 0xY’Y’VVYYUU equivalent to four unsigned char for two pixels. With Y luma for pixel n, Y’ luma for pixel n+1, and U and V are the average chrominance value of both pixels.
# AL::kYUVColorSpace	    10	3	3	Buffer contains triplet on the format 0xVVUUYY, equivalent to three unsigned char
# AL::kRGBColorSpace	    11	3	3	Buffer contains triplet on the format 0xRRGGBB, equivalent to three unsigned char
# AL::kHSYColorSpace	    12	3	3	Buffer contains triplet on the format 0xYYSSHH, equivalent to three unsigned char
# AL::kBGRColorSpace	    13	3	3	Buffer contains triplet on the format 0xBBGGRR, equivalent to three unsigned char
# AL::kYYCbCrColorSpace	    14	2	2	TIFF format, four unsigned characters for two pixels.
# AL::kH2RGBColorSpace	    15	3	3	H from “HSY to RGB” in fake colors.
# AL::kHSMixedColorSpace	16	3	3	HS and (H+S)/2.


##########################################################################
############################### 3D CAMERA ################################
##########################################################################

# 3D camera supported colorspaces

# Parameter ID Name	    ID Value	Number of layers    Number of channels	Description
# AL::kYuvColorSpace	    0	            1	                1               Contains the depth image clamped from 400m to 3.68m scaled in range [0,255], equivalent to one unsigned char
# AL::kRGBColorSpace	    11	            3	                3	            Contains the depth image in false-color (RGB) equivalent to three unsigned char (debug purpose)
# AL::kDepthColorSpace	    17	            2	                1	            Contains the distance in mm from the image plan corrected, equivalent to one unsigned short
# AL::kXYZColorSpace	    19	            12	                3	            Contains the position in meter of the voxel in the camera frame, equivalent to three float
# AL::kDistanceColorSpace	21	            2	                1	            Contains the distance from the camera in mm, equivalent to one unsigned short
# AL::kRawDepthColorSpace	23	            2	                1	            Contains the raw distance in mm from the image plan without correction, equivalent to one unsigned short

# 3D camera supported resolutions

# Parameter ID Name	ID Value	Description
# AL::kQQQQVGA	        8	    Image of 40*30px
# AL::kQQQVGA	        7	    Image of 80*60px
# AL::kQQVGA	        0	    Image of 160*120px
# AL::kQVGA	            1	    Image of 320*240px

# 3D camera supported fps

# Resolution	Supported Frame rate
# AL::kQQQQVGA	from 1 to 20 fps
# AL::kQQQVGA	from 1 to 20 fps
# AL::kQQVGA	from 1 to 20 fps
