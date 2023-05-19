from enum import Enum

class CameraResolution3D(Enum):
    """
    Pepper 3D camera resolutions 

    Resolution            ID Value    local	Gb Ethernet	100Mb Ethernet	WiFi g
    40x30     (QQQQVGA)	    8       30fps	        30fps       30fps	30fps
    80x60     (QQQVGA)	    7       30fps	        30fps	    30fps	30fps
    160x120   (QQVGA)	        0       30fps	        30fps	    30fps	30fps
    320x240   (QVGA)	        1       30fps	        30fps	    30fps	11fps
    """
    R40x30= 8
    R80x60= 7
    R160x120= 0
    R320x240= 1

class CameraResolution2D(Enum):
    """
    Pepper 2D camera resolutions 

    Resolution            ID Value    local	Gb Ethernet	100Mb Ethernet	WiFi g
    40x30     (QQQQVGA)	    8       30fps	        30fps       30fps	30fps
    80x60     (QQQVGA)	    7       30fps	        30fps	    30fps	30fps
    160x120   (QQVGA)	        0       30fps	        30fps	    30fps	30fps
    320x240   (QVGA)	        1       30fps	        30fps	    30fps	11fps
    640x480   (VGA)	        2       30fps	        30fps	    12fps	2.5fps
    1280x960  (4VGA)	        3       29fps	        10fps	    3fps	0.5fps
    2560x1920 (16VGA)
    """
    R40x30= 8
    R80x60= 7
    R160x120= 0
    R320x240= 1
    R640x480= 2
    R1280x960= 3
    R2560x1920= 4

class CameraID(Enum):
    """
    Pepper camera indexes

    Parameter ID Name	    ID Value    Type
    AL::kTopCamera	        0       2D Camera
    AL::kBottomCamera	        1       2D Camera
    AL::kDepthCamera	        2       Reconstructed 3D Sensor
    """
    TOP = 0
    BOTTOM = 1
    DEPTH = 2


class ColorSpace2D(Enum):
    """
    Colorspace: 2D cameras only

    Parameter            ID Name ID Value	Number of layers	Number of channels	Description
    AL::kYuvColorSpace	0	1	1	Buffer only contains the Y (luma component) equivalent to one unsigned char
    AL::kyUvColorSpace	1	1	1	Buffer only contains the U (Chrominance component) equivalent to one unsigned char
    AL::kyuVColorSpace	2	1	1	Buffer only contains the V (Chrominance component) equivalent to one unsigned char
    AL::kRgbColorSpace	3	1	1	Buffer only contains the R (Red component) equivalent to one unsigned char
    AL::krGbColorSpace	4	1	1	Buffer only contains the G (Green component) equivalent to one unsigned char
    AL::krgBColorSpace	5	1	1	Buffer only contains the B (Blue component) equivalent to one unsigned char
    AL::kHsyColorSpace	6	1	1	Buffer only contains the H (Hue component) equivalent to one unsigned char
    AL::khSyColorSpace	7	1	1	Buffer only contains the S (Saturation component) equivalent to one unsigned char
    AL::khsYColorSpace	8	1	1	Buffer only contains the Y (Brightness component) equivalent to one unsigned char
    AL::kYUV422ColorSpace	9	2	2	Native format, 0xY’Y’VVYYUU equivalent to four unsigned char for two pixels. With Y luma for pixel n, Y’ luma for pixel n+1, and U and V are the average chrominance value of both pixels.
    AL::kYUVColorSpace	10	3	3	Buffer contains triplet on the format 0xVVUUYY, equivalent to three unsigned char
    AL::kRGBColorSpace	11	3	3	Buffer contains triplet on the format 0xBBGGRR, equivalent to three unsigned char
    AL::kHSYColorSpace	12	3	3	Buffer contains triplet on the format 0xYYSSHH, equivalent to three unsigned char
    AL::kBGRColorSpace	13	3	3	Buffer contains triplet on the format 0xRRGGBB, equivalent to three unsigned char
    AL::kYYCbCrColorSpace	14	2	2	TIFF format, four unsigned characters for two pixels.
    AL::kH2RGBColorSpace	15	3	3	H from “HSY to RGB” in fake colors.
    AL::kHSMixedColorSpace	16	3	3	HS and (H+S)/2.
    """
    YuvColorSpace= 0
    yUvColorSpace= 1
    yuVColorSpace= 2
    RgbColorSpace= 3
    rGbColorSpace= 4
    rgBColorSpace= 5
    HsyColorSpace= 6
    hSyColorSpace= 7
    hsYColorSpace= 8
    YUV422ColorSpace= 9
    YUVColorSpace= 10
    RGBColorSpace= 11
    HSYColorSpace= 12
    BGRColorSpace= 13
    YYCbCrColorSpace= 14
    H2RGBColorSpace= 15
    HSMixedColorSpace= 16

class ColorSpace3D(Enum):
    """
    Colorspace: 3D cameras only

    Parameter ID  Name	    ID Value Number of layers    Number of channels     Description
    AL::kYuvColorSpace	    0	    1	                        1	            Contains the depth image clamped from 400m to 3.68m scaled in range [0,255], equivalent to one unsigned char
    AL::kRGBColorSpace	    11	    3	                        3	            Contains the depth image in false-color (RGB) equivalent to three unsigned char (debug purpose)
    AL::kDepthColorSpace	17	    2	                        1	            Contains the distance in mm from the image plan corrected, equivalent to one unsigned short
    AL::kXYZColorSpace	    19	    12	                        3	            Contains the position in meter of the voxel in the camera frame, equivalent to three float
    AL::kDistanceColorSpace	21	    2	                        1	            Contains the distance from the camera in mm, equivalent to one unsigned short
    AL::kRawDepthColorSpace	23	    2	                        1	            Contains the raw distance in mm from the image plan without correction, equivalent to one unsigned short
    """
    YuvColorSpace= 0	
    RGBColorSpace= 11	
    DepthColorSpace= 17
    XYZColorSpace= 19
    DistanceColorSpace= 21
    RawDepthColorSpace= 23