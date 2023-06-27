import time
import cv2
import numpy as np

import rospy
from PIL import Image

from perception_utils.bcolors import bcolors
from ObjectDetection.utils import Model

class YOLOV8(Model):
    def __init__(self, model_name='ycb_320', _conf_threshold=0.3, _iou_threshold=0.5, inference_engine='opencv'):

        # instantiate super attribute
        # custom_args = {'engine': inference_engine, 'model_name': model_name, 'conf_thres': conf_thres, 'iou_thres': iou_thres}
        args = self.get_model(model_name,_conf_threshold,_iou_threshold,inference_engine)
        
        super(YOLOV8, self).__init__(args)
        
        # load the model according to config
        assert self.engine in ['opencv', 'onnxruntime'], 'Inference engine not supported'

        self.model = self.load_model()
                
    def get_model(self, model_name,_conf_threshold, _iou_threshold, inference_engine):
        
        self.path_catalog = {
            'robocup_320': {
            'engine' : inference_engine,
            'model_name': model_name,
            '_conf_threshold': _conf_threshold, 
            '_iou_threshold': _iou_threshold, 
            'type': 'onnx',
            'model_path':  'scripts/models/ObjectDetection/YOLOV8/weights/RoboCup/robocup.onnx',
            'classes': 'scripts/models/ObjectDetection/YOLOV8/weights/RoboCup/robocup.txt',
            'input_size': 320,
            },
            'ycb_320': {
            'engine' : inference_engine,
            'model_name': model_name,
            '_conf_threshold': _conf_threshold, 
            '_iou_threshold': _iou_threshold, 
            'type': 'onnx',
            'model_path':  'scripts/models/ObjectDetection/YOLOV8/weights/YCB/yolov8n_320_fp32.onnx',
            'classes': 'scripts/models/ObjectDetection/YOLOV8/weights/YCB/classes.txt',
            'input_size': 320,
            },
            'ycb_640': {
            'engine' : inference_engine,
            'model_name': model_name,
            '_conf_threshold': _conf_threshold, 
            '_iou_threshold': _iou_threshold, 
            'type': 'onnx',
            'model_path': 'scripts/models/ObjectDetection/YOLOV8/weights/YCB/yolov8n_640.onnx',
            'classes': 'scripts/models/ObjectDetection/YOLOV8/weights/YCB/classes.txt',
            'input_size': 640,
            },
             'coco_320': {
            'engine' : inference_engine,
            'model_name': model_name,
            '_conf_threshold': _conf_threshold, 
            '_iou_threshold': _iou_threshold, 
            'type': 'onnx',
            'model_path': 'scripts/models/ObjectDetection/YOLOV8/weights/coco/yolov8n_320.onnx',
            'classes': 'scripts/models/ObjectDetection/YOLOV8/weights/coco/coco.txt',
            'input_size': 320,
            },           
            'receptionist_320': {
            'engine' : inference_engine,
            'model_name': model_name,
            '_conf_threshold': _conf_threshold, 
            '_iou_threshold': _iou_threshold, 
            'type': 'onnx',
            'model_path': 'scripts/models/ObjectDetection/YOLOV8/weights/receptionist/receptionist_320.onnx',
            'classes': 'scripts/models/ObjectDetection/YOLOV8/weights/receptionist/receptionist.txt',
            'input_size': 320,
            },
            'receptionist_640': {
            'engine' : inference_engine,
            'model_name': model_name,
            '_conf_threshold': _conf_threshold, 
            '_iou_threshold': _iou_threshold, 
            'type': 'onnx',
            'model_path': 'scripts/models/ObjectDetection/YOLOV8/weights/receptionist/receptionist_640.onnx',
            'classes': 'scripts/models/ObjectDetection/YOLOV8/weights/receptionist/receptionist.txt',
            'input_size': 640,
            },
            'shoes_320': {
            'engine' : inference_engine,
            'model_name': model_name,
            '_conf_threshold': _conf_threshold, 
            '_iou_threshold': _iou_threshold, 
            'type': 'onnx',
            'model_path': 'scripts/models/ObjectDetection/YOLOV8/weights/shoes/shoes_320v2.onnx',
            'classes': 'scripts/models/ObjectDetection/YOLOV8/weights/shoes/footwear.txt',
            'input_size': 320,
            },
            'clothes_320': {
            'engine' : inference_engine,
            'model_name': model_name,
            '_conf_threshold': _conf_threshold, 
            '_iou_threshold': _iou_threshold, 
            'type': 'onnx',
            'model_path': 'scripts/models/ObjectDetection/YOLOV8/weights/clothes/clothes_320.onnx',
            'classes': 'scripts/models/ObjectDetection/YOLOV8/weights/clothes/clothes.txt',
            'input_size': 320,
            },
            'drinks_320': {
            'engine' : inference_engine,
            'model_name': model_name,
            '_conf_threshold': _conf_threshold, 
            '_iou_threshold': _iou_threshold, 
            'type': 'onnx',
            'model_path': 'scripts/models/ObjectDetection/YOLOV8/weights/drink/drinks_320.onnx',
            'classes': 'scripts/models/ObjectDetection/YOLOV8/weights/drink/drink.txt',
            'input_size': 320,
            },
            'bag_320': {
            'engine' : inference_engine,
            'model_name': model_name,
            '_conf_threshold': _conf_threshold, 
            '_iou_threshold': _iou_threshold, 
            'type': 'onnx',
            'model_path': 'scripts/models/ObjectDetection/YOLOV8/weights/bag/bag_320.onnx',
            'classes': 'scripts/models/ObjectDetection/YOLOV8/weights/bag/bag.txt',
            'input_size': 320,
            }            
        }

        return self.path_catalog[model_name]

    def load_model(self):
        if self.type == 'onnx':
            if self.engine == 'opencv':
                return cv2.dnn.readNetFromONNX(self.model_path)
            elif self.engine == 'onnxruntime':
                import onnxruntime
                return onnxruntime.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            
            rospy.loginfo(
                bcolors.CYAN+"[RoboBreizh - Vision]    Loading Object Detection weights done"+bcolors.ENDC)
            
        else:
            raise Exception('Model type not yet supported')
        
    def get_input_size(self):
        try:
            # Get the input layer
            input_layer = self.model.getLayer(self.model.getLayerId('images'))

            # Get the input size
            input_size = input_layer.params['blobs'][0].shape[1:2]
            assert len(input_size) == 2, 'Invalid input size'
            return int(input_size[0])
        except:
            rospy.loginfo(bcolors.INFO+"[RoboBreizh - YOLOV8] Input layer not detected, input size set to (320,320)"+bcolors.ENDC)
            return 320
        
    def inference(self, image):
        
        time_start = time.time()
        
        image = self.pre_process(image)

        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255, size=(self.input_size, self.input_size), swapRB=True) # blobFromImage does not respect the aspect ratio!
        self.model.setInput(blob)
        output = self.model.forward()

        detections = self.post_process(output, image)
        
        time_end = time.time()
        
        rospy.loginfo("YOLOV8 Object Detection Inference time : " + str(time_end-time_start))

        return detections
    
    def pre_process(self, image):
        # check image format between cv2, numpy or PIL
        if isinstance(image, np.ndarray):
            image = image
        elif isinstance(image, Image.Image):
            image = np.array(image)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise Exception('Image format not supported')

        [height, width, _] = image.shape
        length = max((height, width))
        img = np.zeros((length, length, 3), np.uint8)
        img[0:height, 0:width] = image

        return img
    
    def post_process(self, outputs, image):
        [height, width, _] = image.shape
        length = max((height, width))
        scale = length / self.input_size

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]
        boxes = []
        confidences = []
        class_ids = []
        
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= self._conf_threshold:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                confidences.append(maxScore)
                class_ids.append(maxClassIndex)         

        result_boxes = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.iou_threshold)

        detections = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            # rescale box
            box = [round(box[0] * scale), round(box[1] * scale), round(box[2] * scale), round(box[3] * scale)]
            detection = {
                'class_id': class_ids[index],
                'class_name': self.classes[class_ids[index]],
                'confidence': confidences[index],
                'box': box}
            detections.append(detection)

        return detections
