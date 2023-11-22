import time
import cv2
import numpy as np

import rospkg, rospy
import os
from PIL import Image

from perception_utils.bcolors import bcolors
from ObjectDetection.utils import Model

class YOLOV5(Model):
    def __init__(self, dataset_name='ycb_320', conf_thres=0.3, iou_thres=0.5, inference_engine='opencv'):

        # instantiate super attribute
        custom_args = {'engine': inference_engine, 'dataset_name': dataset_name, 'conf_thres': conf_thres, 'iou_thres': iou_thres}
        args = self.get_model(dataset_name).update(custom_args)
        super(YOLOV5, self).__init__(args)
        
        # load the model according to config
        assert self.engine in ['opencv', 'onnxruntime'], 'Inference engine not supported'

        self.model = self.load_model()

    def get_model(self, dataset_name):
        self.path_catalog = {'ycb_320': {
            'type': 'onnx',
            'model_path':  os.path.join(self.base_path, 'models/ObjectDetection/YOLOV8/weights/YCB/yolov8n_320.onnx'),
            'classes': os.path.join(self.base_path, 'models/ObjectDetection/YOLOV8/weights/YCB/objects.txt'),
            'input_size': 320,
            },
            'ycb_640': {
            'type': 'onnx',
            'model_path':  os.path.join(self.base_path, 'models/ObjectDetection/YOLOV8/weights/YCB/yolov8n_640.onnx'),
            'classes': os.path.join(self.base_path, 'models/ObjectDetection/YOLOV8/weights/YCB/objects.txt'),
            'input_size': 640,
            },
        }

        return self.path_catalog[dataset_name]

    def load_model(self):
        # load classes
        if self.type == 'onnx':
            if self.engine == 'opencv':
                return cv2.dnn.readNetFromONNX(self.model_path)
            elif self.engine == 'onnxruntime':
                import onnxruntime
                return onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        else:
            raise Exception('Model type not yet supported')
        
    def get_input_size(self):
        try:
            # Get the input layer
            input_layer = self.model.getLayer(self.model.getLayerId('images'))

            # Get the input size
            input_size = input_layer.params['blobs'][0].shape[1:2]
            assert len(input_size) == 2, 'Invalid input size'
            return input_size
        except:
            rospy.loginfo(bcolors.INFO+"[RoboBreizh - YOLOV8] Input layer not detected, input size set to (320,320)"+bcolors.ENDC)
            return (320, 320)
        
    def inference(self, image):
        image = self.pre_process(image)

        blob = cv2.dnn.blobFromImage(image) # blobFromImage does not respect the aspect ratio!
        self.model.setInput(blob)
        output = self.model.forward()

        detections = self.post_process(output, image)

        return detections
    
    def pre_process(self, image):
        # check image format between cv2, numpy or PIL
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise Exception('Image format not supported')
        input_size = self.get_input_size()
        image = cv2.resize(image, input_size)

        [height, width, _] = image.shape
        length = max((height, width))
        img = np.zeros((length, length, 3), np.uint8)
        img[0:height, 0:width] = image
        scale = length / res

        return image
    
    def post_process(self, image, outputs):
        image_height, image_width, _ = image.shape
        boxes = []
        confidences = []
        class_ids = []
        for output in outputs[0, 0]:
            confidence = output[2]
            if confidence > self.conf_threshold:
                class_id = int(output[1])
 
                box = output[3:7] * np.array([image_width, image_height, image_width, image_height])
                boxes.append(box.astype('int'))
                confidences.append(float(confidence))
                class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences,  score_threshold=self.conf_threshold, nms_threshold=self.iou_threshold)

        detections = []
        for i in range(len(indices)):
            index = indices[i]
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'class_name': self.classes[class_ids[index]],
                'confidence': confidences[index],
                'box': box}
            detections.append(detection)

        return detections
