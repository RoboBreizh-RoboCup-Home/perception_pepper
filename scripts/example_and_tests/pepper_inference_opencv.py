
import time
import cv2
import numpy as np
from cv_bridge import CvBridge
import rospy
import message_filters
from PIL import Image
from sensor_msgs.msg import Image as Image2

import cv2
import qi 

from yolov8 import YOLOv8
from yolov8.utils import draw_bounding_box_opencv
from yolov8.utils import class_names as CLASSES
import argparse
from Naoqi_camera import NaoqiCamera
from tflite_runtime.interpreter import Interpreter
from yolov8.utils import xywh2xyxy, nms, draw_detections

class Detector():
    def __init__(self, model, res):
        rospy.init_node('YoloV8', anonymous=True)

        self.bridge = CvBridge()
        self.res = int(res)
        self.session = qi.Session()
        self.session.connect("tcp://127.0.0.1:9559")

        self.conf_threshold = 0.3
        self.iou_threshold = 0.5

        self.cam = NaoqiCamera(res, "top")

        self.model = "./models/yolov8/"+model
        

        if "tflite" in model:
            self.tflit_detector = self.init_tflite(self.model)
            self.pub_tflite = rospy.Publisher(
                'yolov8_detector_tflite', Image2, queue_size=1)
            use_tflite= True
        else:
            self.yolov8_detector = YOLOv8(conf_thres=self.conf_threshold, iou_thres=self.iou_threshold)

            self.cv2_detector = cv2.dnn.readNetFromONNX(self.model)
            self.yolov8_detector.initialize_model(self.model)
        
            self.pub_onnx = rospy.Publisher(
                    'yolov8_detector_onnx', Image2, queue_size=1)

            self.pub_cv2 = rospy.Publisher(
                    'yolov8_detector_cv', Image2, queue_size=1)
            use_tflite= False

        # spin
        print("Waiting for image topics...")
        while not rospy.is_shutdown():
            self.image_callback(use_tflite)
        # rospy.spin()

    def init_tflite(self, model):
        # Load the TFLite model and allocate tensors.
        self.tflite_interpreter = Interpreter(model_path=model)
        self.tflite_interpreter.allocate_tensors()

    def extract_boxes(self, predictions, original_shape, input_height, input_width):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes, original_shape, input_height, input_width)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes, original_shape, input_height, input_width):
        img_height, img_width = original_shape
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([img_width, img_height, img_width, img_height])
        return boxes
    
    def detect_tflite(self, image):
        time_1 = time.time()
        [height, width, _] = image.shape
        length = max((height, width))
        scale = length / self.res

        # Get input and output tensors.
        self.input_details = self.tflite_interpreter.get_input_details()
        self._input_type = self.input_details[0]['dtype']
        self.output_details = self.tflite_interpreter.get_output_details()

        input_height =  self.tflite_interpreter.get_input_details()[0]['shape'][1]
        input_width =  self.tflite_interpreter.get_input_details()[0]['shape'][2]

        # Resize and pad the image to keep the aspect ratio and fit the expected size.
      
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (input_width, input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        self.tflite_interpreter.allocate_tensors()

        # Run inference 
        self.tflite_interpreter.set_tensor(self.input_details[0]['index'],
                                    input_tensor.astype(self._input_type))
        self.tflite_interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        outputs = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])

        predictions = np.squeeze(outputs[0]).T

        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions, original_shape=(height, width), input_height=input_height, input_width=input_width)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)
        boxes = boxes[indices]
        scores = scores[indices]
        class_ids = class_ids[indices]
        mask_alpha = 0.4

        img=draw_detections(image, boxes, scores, class_ids, mask_alpha)
        
        time_2=time.time()

        print("Detected class ids: ", class_ids)
        print("Detection time TfLite:", time_2 - time_1)
        print("Object detected TfLite: ", len(boxes))

        return img
    
    def detect_onnx(self, image):
        time_1 = time.time()

        boxes, scores, class_ids = self.yolov8_detector(image)

        combined_img = self.yolov8_detector.draw_detections(image)
        time_2=time.time()
        print("Detected class ids: ", class_ids)
        print("Detection time ONNX:", time_2 - time_1)
        print("Object detected ONNX: ", len(boxes))

        return combined_img

    def detect_opencv(self, orig_image, res):
        time_1 = time.time()

        [height, width, _] = orig_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = orig_image
        scale = length / res

        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(res, res))
        self.cv2_detector.setInput(blob)
        outputs = self.cv2_detector.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'class_name': CLASSES[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale}
            detections.append(detection)
            img = draw_bounding_box_opencv(orig_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                            round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

        time_2=time.time()
        print("Detection time OPENCV:", time_2 - time_1)
        print("Object detected OPENCV: ", len(detections))
        return img

    def image_callback(self, use_tflite):
        frame = self.cam.get_image('cv2')
        
        if use_tflite is False:
            onnx_out = self.detect_onnx(frame)
            print("##############################")
            opencv_out = self.detect_opencv(frame, self.res)
            print("##############################")

            ros_image_yolo_cv = self.bridge.cv2_to_imgmsg(opencv_out, "rgb8")

            self.pub_cv2.publish(ros_image_yolo_cv)
        
            ros_image_yolo_onnx = self.bridge.cv2_to_imgmsg(onnx_out, "rgb8")
            self.pub_onnx.publish(ros_image_yolo_onnx)
        else:
            tflite_out = self.detect_tflite(frame)
            ros_image_yolo_tflite = self.bridge.cv2_to_imgmsg(tflite_out, "rgb8")
            self.pub_tflite.publish(ros_image_yolo_tflite)


if __name__ == '__main__':
    # get arg 1 and 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8n_ycb.onnx', help='model path')
    parser.add_argument('--res', type=str, default='640', help='resolution')

    args = parser.parse_args()
    model = args.model
    res = args.res
    print("Starting detection with args: \n model: ", model, "\n resolution: ", res, "\n")
    Detector(model, res)
