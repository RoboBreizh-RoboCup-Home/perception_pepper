
import time
import cv2
import numpy as np
from cv_bridge import CvBridge
import rospy
from PIL import Image
from sensor_msgs.msg import Image as Image2

import cv2
import qi 

import argparse
from Naoqi_camera import NaoqiSingleCamera

with open("/home/maelic/Documents/robocup2023/perception_pepper/scripts/models/ObjectDetection/YOLOV8/weights/robocup/robocup.txt", 'r') as f:
    CLASSES = f.read().splitlines()

import numpy as np
import cv2

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(CLASSES), 3))

class Detector():
    def __init__(self, model, res):
        rospy.init_node('YoloV8', anonymous=True)

        self.bridge = CvBridge()
        self.res = int(res)
        self.session = qi.Session()
        self.session.connect("tcp://192.168.0.103:9559")

        self.conf_threshold = 0.3
        self.iou_threshold = 0.5

        self.cam = NaoqiSingleCamera(ip="192.168.0.103")

        self.model = model
        
        self.cv2_detector = cv2.dnn.readNetFromONNX(self.model)

        self.pub_cv2 = rospy.Publisher(
                'yolov8_detector_cv', Image2, queue_size=1)
        use_tflite= False

        # spin
        print("Waiting for image topics...")
        while not rospy.is_shutdown():
            self.image_callback(use_tflite)
        # rospy.spin()

    def draw_bounding_box_opencv(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f'{CLASSES[class_id]} ({confidence:.2f})'
        color = colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def xywh2xyxy(x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def extract_boxes(self, predictions, original_shape, input_height, input_width):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes, original_shape, input_height, input_width)

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes, original_shape, input_height, input_width):
        img_height, img_width = original_shape
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([img_width, img_height, img_width, img_height])
        return boxes
    

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
            img = self.draw_bounding_box_opencv(orig_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                            round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

        time_2=time.time()
        print("Detection time OPENCV:", time_2 - time_1)
        print("Object detected OPENCV: ", len(detections))
        return img

    def image_callback(self, use_tflite):
        frame = self.cam.get_image('cv2')
    
        print("##############################")
        opencv_out = self.detect_opencv(frame, self.res)
        print("##############################")

        ros_image_yolo_cv = self.bridge.cv2_to_imgmsg(opencv_out, "rgb8")

        self.pub_cv2.publish(ros_image_yolo_cv)


if __name__ == '__main__':
    # get arg 1 and 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8n_ycb.onnx', help='model path')
    parser.add_argument('--res', type=str, default='320', help='resolution')

    args = parser.parse_args()
    model = args.model
    res = args.res
    print("Starting detection with args: \n model: ", model, "\n resolution: ", res, "\n")
    Detector(model, res)
