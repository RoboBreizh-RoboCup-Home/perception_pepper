from perception_pepper.perception_utils.bcolors import bcolors
import math
import cv2

class Chair:
    def __init__(self):
        self.isEmpty = True
        self.xyxy = [0, 0, 0, 0]

def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
            
    return iou


def seat_person_mid_point_distance(boxA, boxB):
    
    start_x_boxA = boxA[0]
    start_y_boxA = boxA[1]
    end_x_boxA = boxA[2]
    end_y_boxA = boxA[3]
    
    mid_point_x_boxA = int((start_x_boxA + end_x_boxA) /2)
    mid_point_y_boxA = int((start_y_boxA + end_y_boxA) /2)
    
    mid_point_boxA = [mid_point_x_boxA,mid_point_y_boxA]
    
    start_x_boxB = boxB[0]
    start_y_boxB = boxB[1]
    end_x_boxB = boxB[2]
    end_y_boxB = boxB[3]
     
    mid_point_x_boxB = int((start_x_boxB + end_x_boxB) /2)
    mid_point_y_boxB = int((start_y_boxB + end_y_boxB) /2)
    
    mid_point_boxB = [mid_point_x_boxB,mid_point_y_boxB]

    distance = math.dist(mid_point_boxA, mid_point_boxB)
    
    return distance
    
def range_overlap(a_min, a_max, b_min, b_max, margin):
    """Neither range is completely greater than the other"""
    return (a_min + margin <= b_max) and (b_min + margin <= a_max)


def position_in_couch_available(xyxy_lastHuman, couch):
    top_human = xyxy_lastHuman[0]
    left_human = xyxy_lastHuman[1]
    bottom_human = xyxy_lastHuman[2]
    right_human = xyxy_lastHuman[3]

    top_couch = couch.xyxy[0]
    left_couch = couch.xyxy[1]
    bottom_couch = couch.xyxy[2]
    right_couch = couch.xyxy[3]

    x_center_couch = right_couch - left_couch
    x_center_human = right_human - left_human

    if x_center_couch < x_center_human:  # human on the right
        return [top_couch, left_couch, bottom_couch, right_couch - right_couch / 3]

    else:  # human on the left
        return [top_couch, left_couch, bottom_couch, right_couch + right_couch / 3]


def intersecting(rect1, rect2, margin):
    """
    Returns whether two rectangles overlap.
    rect: [xmin, ymin, xmax, ymax]
    :param rect1:
    :param rect2:
    :return:
    """
    x1min, y1min, x1max, y1max = rect1[0], rect1[1], rect1[2], rect1[3]
    x2min, y2min, x2max, y2max = rect2[0], rect2[1], rect2[2], rect2[3]
    return range_overlap(x1min, x1max, x2min, x2max, margin) and range_overlap(
        y1min, y1max, y2min, y2max, margin
    )

def has_chairs_coco_couch(cv_image, classes, detections, imageWidth, imageHeight):
    width = imageWidth
    height = imageHeight

    arr_empty_chairs = []
    arr_taken_chairs = []
    arr_persons = []

    # ------ Human
    for detection in detections:
        classID = int(detection[1])
        classe = classes[classID]
        if classe == "person":
            left = int(detection[3] * width)
            top = int(detection[4] * height)
            right = int(detection[5] * width)
            bottom = int(detection[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            xyxy = [top, left, bottom, right]

            arr_persons.append(xyxy)  # xyxy
            
            cv2.rectangle(cv_image, (left,top), (right, bottom), (0,0,255), 2)

    # ------ Chair
    for detection in detections:
        classID = int(detection[1])
        classe = classes[classID]
        if classe == "chair" or classe == "couch":
            chair = Chair()
            # detection[len(detection) - 1][1] = classe
            left = int(detection[3] * width)
            top = int(detection[4] * height)
            right = int(detection[5] * width)
            bottom = int(detection[6] * height)
            
            cv2.rectangle(cv_image, (left,top), (right, bottom), (0,255,255), 2)

            if left < 0:
                left = 0
            if top < 0:
                top = 0
            chair.xyxy = [top, left, bottom, right]

            margin = 20

            # special case : coach
            bNeedCheckTwoSeat = False
            counterPers = 0

            if classe == "couch":
                bNeedCheckTwoSeat = True

            for person in arr_persons:
                if intersecting(person, chair.xyxy, margin):
                    if bNeedCheckTwoSeat == True:  # cas sofa
                        counterPers = counterPers + 1
                        xyxy_lastHuman = person
                        if counterPers == 2:
                            chair.isEmpty = False
                            break
                    else:  # cas classique
                        chair.isEmpty = False
                        break
                    
            if chair.isEmpty:
                if bNeedCheckTwoSeat == False:
                    arr_empty_chairs.append(chair)  # Chair instances
                else:
                    if counterPers == 0:
                        arr_empty_chairs.append(chair)  # Chair instances
                    elif counterPers == 1:  # cas merdique
                        chair.xyxy = position_in_couch_available(xyxy_lastHuman, chair)
                        arr_empty_chairs.append(chair)  # Chair instances

            else:
                arr_taken_chairs.append(chair)  # Chair instances

    print("     -->  person detected : " + str(len(arr_persons)))
    print("     -->  chair/cought FREE detected : " + str(len(arr_empty_chairs)))
    print("     -->  chair/cought TAKEN detected : " + str(len(arr_taken_chairs)))
    
    return cv_image, arr_persons, arr_empty_chairs, arr_taken_chairs

