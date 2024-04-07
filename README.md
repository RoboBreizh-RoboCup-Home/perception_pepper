# RoboBreizh Perception Package

This package contains models and ROS2 nodes to handle different sorts of image processing required by the tasks of the RoboCup@Home competition. This repository uses the Pepper robot image processing (through Naoqi) as a backend but it can be easily customized to any robot with an RGB-D or RGB camera.


## Overview

The main models used are:

1. AgeGenderDetection: From an RGB image of a person's face, detect the age and gender (M/F). 
There are two models available: 
    1. A simple CNN model taken from [the awesome implementation for Android by Shubham Panchal](https://github.com/shubham0204/Age-Gender_Estimation_TF-Android). This model inference uses the TFLite interpreter.
    2. A Caffe implementation of another CNN [by eveningglow](https://github.com/eveningglow/age-and-gender-classification/) which is older and performs significantly worse.

2. ColourDetection: get back the main color component of an RGB image. Here, we use it to detect clothes' colors by cropping the input image on the cloth detection. This is a custom lightweight implementation based on a kmeans algorithm (inference with OpenCV).

3. FaceDetection: detect all human faces from an input RGB image. The model used here is a lightweight YuNet, [taken from the official model repository of OpenCV](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet). The model perform very good in full precision.

4. GlassesDetection: This is a custom implementation based on [the one from TianxingWu](https://github.com/TianxingWu/realtime-glasses-detection) which is a simple yet efficient glasses detector based on face landmarks.

5. ObjectDetection: We use different state-of-the-art models to detect objects based on input RGB images. The main model used is YOLOV8, from [the ultralytics implementation](https://github.com/ultralytics/ultralytics). This model is fine-tuned on a few-shot dataset of the objects present at the robocup, for a description of the data collection and fine-tuning please see [our dedicated paper](https://link.springer.com/chapter/10.1007/978-3-031-55015-7_31). Other models used are an SSD Inception V2, trained on COCO dataset (80 classes) and an SSD MobileNetV2 trained on OpenImageV4 (600 classes). The MobileNet model is used to detect a wide range of objects in the wild, with worse accuracy than YOLOV8.

6. PoseDetection: For pose detection, we use two MoveNet models, from the [original TFLite implementation](https://www.tensorflow.org/hub/tutorials/movenet). The MoveNet Multipose is used to detect the pose of multiple persons at a time, with higher inference time while the MoveNet Lightning is an ultra-fast detector that can only detect the pose of a single human at a time.

## Usage

You can use the different models with ROS2 by calling the different ROS [nodes](perception_pepper/ros_node). Be careful though, not all nodes have been ported to ROS2 yet. To install the package, clone it in a ROS2 workspace (assuming ros2_ws here) and run:

```
cd ros2_ws/src
git clone https://github.com/RoboBreizh-RoboCup-Home/perception_pepper.git
colcon build --symlink-install --packages-select perception_pepper
source ./install/local_setup.bash
```

As an example, you can run some [out-of-the-box demos](perception_pepper/example_and_tests) for the pose detection and features detection (age + gender + clothes) as follows:

```
ros2 run perception_pepper pose_demo
ros2 run perception_pepper features_demo
```


## Citations

If you use this project, please consider citing:

```
@incollection{buche2023robocup,
  title={RoboCup@ Home SSPL Champion 2023: RoboBreizh, a Fully Embedded Approach},
  author={Buche, C{\'e}dric and Neau, Ma{\"e}lic and Ung, Thomas and Li, Louis and Wang, Sinuo and Bono, C{\'e}dric Le},
  booktitle={Robot World Cup},
  pages={374--385},
  year={2023},
  publisher={Springer}
}

```
