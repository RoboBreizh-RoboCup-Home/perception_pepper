#! /usr/bin/env python
# ----------------------------------------------------------------------------
# Authors  : Cedric BUCHE (buche@enib.fr)
# Created Date: 2022
# ---------------------------------------------------------------------------


import rospkg
import rospy
import operator
import time
import perception_utils.bcolors


###############################################################
def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


######################################################################


def range_overlap(a_min, a_max, b_min, b_max, margin):
    """Neither range is completely greater than the other"""
    return (a_min + margin <= b_max) and (b_min + margin <= a_max)


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


######################################################################
def group_similar_objects_oid(detection):
    classID = int(detection[1])
    # hand bag (405), (Luggage and bag) 554
    if classID == 405 or classID == 554:
        classe = "Bag"
    if classID == 328:  # Plastic bag for trash bag?
        classe = "Trash bag"
    if classID == 312:  # change plate to dish
        classe = "Dish"
    if classID == 196:  # change flying disc to disk
        classe = "Disk"
    if classID == 293 or classID == 237:  # Bowl(293), Mixing bowl (237)
        classe = "Bowl"
        # Coffee cup(179), Measuring cup(401)
    if classID == 179 or classID == 401:
        classe = "Cup"
    if classID == 191:  # change paper towel to napkin
        classe = "Napkin"
    if classID == 325 or classID == 285:  # Knife 285, Kitchen Knife 325
        classe = "Knife"
    if classID == 395:  # Picnic Basket
        classe = "Basket"
    if classID == 444:  # Waste container
        classe = "Rubbish Bin"
    else:
        classe = self.classes[classID - 1]
    return classe


################################################################################
# Originally 601 classes


def simplify_classname_oid(classe):
    classes = {
        "Plastic bag": "Bag",
        "Handbag": "Bag",
        "Luggage and bags": "Bag",
        "Backpack": "Bag",
        "Suitcase": "Bag",
        "Briefcase": "Bag",
        "Table": "Table",
        "Billiard table": "Table",
        "Coffee table": "Table",
        "Kitchen & dining room table": "Table",
        "Desk": "Table",
        "Clothing": "Clothing",
        "Shirt": "Clothing",
        "Sock": "Clothing",
        "Trousers": "Clothing",
        "Shorts": "Clothing",
        "Dress": "Clothing",
        "Skirt": "Clothing",
        "Miniskirt": "Clothing",
        "Jeans": "Clothing",
        "Coat": "Clothing",
        "Suit": "Clothing",
        "Swimwear": "Clothing",
        "Scarf": "Clothing",
        "Jacket": "Clothing",
        "Footwear": "Clothing",
        "Hat": "Clothing",
        "Sandal": "Clothing",
        "Boot": "Clothing",
        "Brassiere": "Clothing",
        "Tie": "Clothing",
        "Cowboy hat": "Clothing",
        "Fedora": "Clothing",
        "Roller skates": "Clothing",
        "Sun hat": "Clothing",
        "Lifejacket": "Clothing",
        "High heels": "Clothing",
        "Kitchen utensil": "Utensils",
        "Fork": "Utensils",
        "Knife": "Utensils",
        "Frying pan": "Utensils",
        "Plate": "Utensils",
        "Kitchen knife": "Utensils",
        "Wine glass": "Utensils",
        "Spoon": "Utensils",
        "Chopsticks": "Utensils",
        "Drinking straw": "Utensils",
        "Mixing bowl": "Tableware",
        "Bowl": "Tableware",
        "Serving tray": "Tableware",
        "Mug": "Tableware",
        "Coffee cup": "Tableware",
        "Teapot": "Tableware",
        "Tableware": "Tableware",
        "Bottle": "Tableware",
        "Saucer": "Tableware",
        "Kettle": "Tableware",
        "Platter": "Tableware",
        "Drink": "Drink",
        "Juice": "Drink",
        "Milk": "Drink",
        "Coffee": "Drink",
        "Beer": "Drink",
        "Dairy": "Drink",
        "Tea": "Drink",
        "Cocktail": "Drink",
        "Wine": "Drink",
        "Food": "Food",
        "Croissant": "Food",
        "Doughnut": "Food",
        "Hot dog": "Food",
        "Fast food": "Food",
        "Popcorn": "Food",
        "Cheese": "Food",
        "Muffin": "Food",
        "Cookie": "Food",
        "Dessert": "Food",
        "French fries": "Food",
        "Baked goods": "Food",
        "Pasta": "Food",
        "Pizza": "Food",
        "Sushi": "Food",
        "Bread": "Food",
        "Ice cream": "Food",
        "Salad": "Food",
        "Sandwich": "Food",
        "Pastry": "Food",
        "Waffle": "Food",
        "Pancake": "Food",
        "Burrito": "Food",
        "Snack": "Food",
        "Taco": "Food",
        "Hamburger": "Food",
        "Cake": "Food",
        "Honeycomb": "Food",
        "Pretzel": "Food",
        "Bagel": "Food",
        "Guacamole": "Food",
        "Pretzel": "Food",
        "sandwich": "Food",
        "hot dog": "Food",
        "pizza": "Food",
        "donut": "Food",
        "cake": "Food",
        "Candy": "Food",
        "Apple": "Fruit",
        "Fruit": "Fruit",
        "Grape": "Fruit",
        "Tomato": "Fruit",
        "Lemon": "Fruit",
        "Banana": "Fruit",
        "Orange": "Fruit",
        "Coconut": "Fruit",
        "Mango": "Fruit",
        "Pineapple": "Fruit",
        "Grapefruit": "Fruit",
        "Pomegranate": "Fruit",
        "Watermelon": "Fruit",
        "Strawberry": "Fruit",
        "Peach": "Fruit",
        "Cantaloupe": "Fruit",
        "apple": "Fruit",
        "banana": "Fruit",
        "orange": "Fruit",
        "broccoli": "Vegetable",
        "carrot": "Vegetable",
        "Human face": "Person",
        "Human body": "Person",
        "Man": "Person",
        "Woman": "Person",
        "Person": "Person",
        "Boy": "Person",
        "Girl": "Person",
        "Human head": "Person",
        "Chair": "Seat",
        "Sofa bed": "Seat",
        "Studio couch": "Seat",
        "Couch": "Seat",
        "Loveseat": "Seat",
        "Stool": "Seat",
        "Toaster": "Cookware",
        "Oven": "Cookware",
        "Spatula": "Cookware",
        "Can opener": "Cookware",
        "Cutting board": "Cookware",
        "Blender": "Cookware",
        "Slow cooker": "Cookware",
        "Paper towel": "Cookware",
        "Gas stove": "Cookware",
        "Salt and pepper shakers": "Cookware",
        "Food processor": "Cookware",
        "Wood-burning stove": "Cookware",
        "Cocktail shaker": "Cookware",
        "Bottle opener": "Cookware",
        "Frying pan": "Cookware",
        "Waffle iron": "Cookware",
        "Cooking spray": "Cookware",
        "Measuring cup": "Cookware",
        "Coffeemaker": "Cookware",
        "Wok": "Cookware",
        "Pizza cutter": "Cookware",
        "Beaker": "Cookware",
        "Microwave oven": "Cookware",
        "Grinder": "Cookware",
        "Spice rack": "Cookware",
        "Cake stand": "Cookware",
        "Pressure cooker": "Cookware",
        "Kitchen appliance": "Cookware",
        "Dishwasher": "Cookware",
    }

    return classes[classe]


######################################################################
# check "bag"
######################################################################


def has_bags_oid(classes, detections, imageWidth, imageHeight):
    width = imageWidth
    height = imageHeight

    arr_bags = []
    for detection in detections:
        classID = int(detection[1])
        classe = classes[classID - 1]
        if (
            classe == "Plastic bag"
            or classe == "Handbag"
            or classe == "Luggage and bags"
            or classe == "Backpack"
            or classe == "Suitcase"
            or classe == "Briefcase"
        ):
            left = int(detection[3] * width)
            top = int(detection[4] * height)
            right = int(detection[5] * width)
            bottom = int(detection[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            xyxy = [top, left, bottom, right]

            arr_bags.append(xyxy)  # xyxy

    return arr_bags


def has_bags_coco(classes, detections, imageWidth, imageHeight):
    width = imageWidth
    height = imageHeight

    arr_bags = []
    for detection in detections:
        classID = int(detection[1])
        classe = classes[classID]
        if classe == "backpack" or classe == "handbag" or classe == "suitcase":
            left = int(detection[3] * width)
            top = int(detection[4] * height)
            right = int(detection[5] * width)
            bottom = int(detection[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            xyxy = [top, left, bottom, right]

            # print("                      " + str(classe) + " " + str(xyxy))

            arr_bags.append(xyxy)  # xyxy

    # remove duplicates
    new_k = []
    for elem in arr_bags:
        if elem not in new_k:
            new_k.append(elem)
    k = new_k
    # print(k)

    return k


#########################################################################################
def has_bag_oid_coco(
    classes, newDetections2_oid, classes_coco, newDetections2_coco, width, height
):
    arr_bags_coco = []
    arr_bags_oid = []
    arr_bags = []

    arr_bags_coco = has_bags_coco(classes_coco, newDetections2_coco, width, height)
    arr_bags_oid = has_bags_oid(classes, newDetections2_oid, width, height)

    # may be remover double ?

    for detection_oid in arr_bags_oid:
        arr_bags.append(detection_oid)
    for detection_coco in arr_bags_coco:
        arr_bags.append(detection_coco)

    return arr_bags


#########################################################################################


class Chair:
    def __init__(self):
        self.isEmpty = True
        self.xyxy = [0, 0, 0, 0]


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


def has_chairs_couch_oid(classes, detections, imageWidth, imageHeight):
    width = imageWidth
    height = imageHeight

    arr_empty_chairs = []
    arr_taken_chairs = []
    arr_persons = []

    # ------ Human
    for detection in detections:
        classID = int(detection[1])
        classe = classes[classID - 1]
        if (
            classe == "Human face"
            or classe == "Human body"
            or classe == "Man"
            or classe == "Woman"
            or classe == "Person"
            or classe == "Boy"
            or classe == "Girl"
            or classe == "Human head"
        ):
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

    # ------ Chair
    for detection in detections:
        classID = int(detection[1])
        classe = classes[classID - 1]
        if (
            classe == "Chair"
            or classe == "Sofa bed"
            or classe == "Studio couch"
            or classe == "Couch"
            or classe == "Loveseat"
            or classe == "Stool"
        ):
            chair = Chair()
            # detection[len(detection) - 1][1] = classe
            left = int(detection[3] * width)
            top = int(detection[4] * height)
            right = int(detection[5] * width)
            bottom = int(detection[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            chair.xyxy = [top, left, bottom, right]

            margin = 20

            # special case : coach
            bNeedCheckTwoSeat = False
            counterPers = 0

            if (
                classe == "Sofa bed"
                or classe == "Studio couch"
                or classe == "Couch"
                or classe == "Loveseat"
            ):
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

    return arr_persons, arr_empty_chairs, arr_taken_chairs


def has_shoes_on_drink_coco(classes, detections, imageWidth, imageHeight):
    width = imageWidth
    height = imageHeight

    arr_empty_shoes = []
    arr_taken_shoes = []
    arr_persons = []
    arr_empty_drink = []
    arr_taken_drink = []

    result_dictionnaire = {}
    dictionnaire = {}

    margin_shoes = 5
    margin_drink = 10
    indexHuman = 0
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

            person = [top, left, bottom, right]
            right_person = right
            indexHuman = indexHuman + 1
            bPersonShoesOn = False
            bPersonDrinkOn = False

            for detection2 in detections:
                classID = int(detection2[1])
                classe = classes[classID]
                if classe == "cup" or classe == "wine glass" or classe == "bottle":
                    left = int(detection2[3] * width)
                    top = int(detection2[4] * height)
                    right = int(detection2[5] * width)
                    bottom = int(detection2[6] * height)
                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0

                    bottle = [top, left, bottom, right]
                    if intersecting(person, bottle, margin_drink):
                        bPersonDrinkOn = True

            footwearCounter = 0
            for detection3 in detections:
                classID = int(detection3[1])
                classe = classes[classID]
                if classe == "boot":
                    left = int(detection3[3] * width)
                    top = int(detection3[4] * height)
                    right = int(detection3[5] * width)
                    bottom = int(detection3[6] * height)
                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0

                    footwear = [top, left, bottom, right]
                    if intersecting(person, footwear, margin_shoes):
                        footwearCounter = footwearCounter + 1
                    if footwearCounter == 2:
                        bPersonShoesOn = True
            text = (
                "Person."
                + str(indexHuman)
                + ".Drink."
                + str(bPersonDrinkOn)
                + ".ShoesOn."
                + str(bPersonShoesOn)
            )
            dictionnaire[text] = int(right_person)

    if len(dictionnaire) != 0:
        sorted_tuples = sorted(dictionnaire.items(), key=operator.itemgetter(1))
        for k, v in sorted_tuples:
            result_dictionnaire[k] = v

    return result_dictionnaire


def has_shoes_on_drink_oid(classes, detections, imageWidth, imageHeight):
    width = imageWidth
    height = imageHeight

    arr_empty_shoes = []
    arr_taken_shoes = []
    arr_persons = []
    arr_empty_drink = []
    arr_taken_drink = []

    result_dictionnaire = {}
    dictionnaire = {}

    margin_shoes = 5
    margin_drink = 10
    indexHuman = 0
    # ------ Human
    for detection in detections:
        classID = int(detection[1])
        classe = classes[classID - 1]
        if (
            classe == "Human face"
            or classe == "Human body"
            or classe == "Man"
            or classe == "Woman"
            or classe == "Person"
            or classe == "Boy"
            or classe == "Girl"
            or classe == "Human head"
        ):
            left = int(detection[3] * width)
            top = int(detection[4] * height)
            right = int(detection[5] * width)
            bottom = int(detection[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0

            person = [top, left, bottom, right]
            right_person = right
            indexHuman = indexHuman + 1
            bPersonShoesOn = False
            bPersonDrinkOn = False

            for detection2 in detections:
                classID = int(detection2[1])
                classe = classes[classID - 1]
                if classe == "Bottle":
                    left = int(detection2[3] * width)
                    top = int(detection2[4] * height)
                    right = int(detection2[5] * width)
                    bottom = int(detection2[6] * height)
                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0

                    bottle = [top, left, bottom, right]
                    if intersecting(person, bottle, margin_drink):
                        bPersonDrinkOn = True

            footwearCounter = 0
            for detection3 in detections:
                classID = int(detection3[1])
                classe = classes[classID - 1]
                if classe == "Footwear":
                    left = int(detection3[3] * width)
                    top = int(detection3[4] * height)
                    right = int(detection3[5] * width)
                    bottom = int(detection3[6] * height)
                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0

                    footwear = [top, left, bottom, right]
                    if intersecting(person, footwear, margin_shoes):
                        footwearCounter = footwearCounter + 1
                    if footwearCounter == 2:
                        bPersonShoesOn = True

            text = (
                "Person."
                + str(indexHuman)
                + ".Drink."
                + str(bPersonDrinkOn)
                + ".ShoesOn."
                + str(bPersonShoesOn)
            )
            dictionnaire[text] = int(right_person)

    if len(dictionnaire) != 0:
        sorted_tuples = sorted(dictionnaire.items(), key=operator.itemgetter(1))
        for k, v in sorted_tuples:
            result_dictionnaire[k] = v

    return result_dictionnaire


def has_shoes_on_drink_oid_coco(
    classes, detections, classes_coco, detections_coco, imageWidth, imageHeight
):
    width = imageWidth
    height = imageHeight

    arr_empty_shoes = []
    arr_taken_shoes = []
    arr_persons_oid = []
    arr_persons_coco = []
    arr_bottles_oid = []
    arr_bottles_coco = []
    arr_shoes_oid = []
    arr_shoes_coco = []
    arr_empty_drink = []
    arr_taken_drink = []

    result_dictionnaire = {}
    dictionnaire = {}

    margin_shoes = 5
    margin_drink = 10
    indexHuman = 0

    # ------ Human OID
    for detection in detections:
        classID = int(detection[1])
        classe_oid = classes[classID - 1]
        if classe_oid == "Man" or classe_oid == "Woman":
            left = int(detection[3] * width)
            top = int(detection[4] * height)
            right = int(detection[5] * width)
            bottom = int(detection[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0

            person = [top, left, bottom, right]
            arr_persons_oid.append(person)

    # ------ Human COCO
    for detection in detections_coco:
        classID = int(detection[1])
        classe = classes_coco[classID]
        if classe == "person":
            left = int(detection[3] * width)
            top = int(detection[4] * height)
            right = int(detection[5] * width)
            bottom = int(detection[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0

            person = [top, left, bottom, right]
            arr_persons_coco.append(person)

    # ------ Bottle OID
    for detection2 in detections:
        classID = int(detection2[1])
        classe = classes[classID - 1]
        if classe == "Bottle":
            left = int(detection2[3] * width)
            top = int(detection2[4] * height)
            right = int(detection2[5] * width)
            bottom = int(detection2[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0

            bottle = [top, left, bottom, right]
            arr_bottles_oid.append(bottle)

    # ------ Shoes OID
    for detection3 in detections:
        classID = int(detection3[1])
        classe = classes[classID - 1]
        if classe == "Footwear":
            left = int(detection3[3] * width)
            top = int(detection3[4] * height)
            right = int(detection3[5] * width)
            bottom = int(detection3[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0

            footwear = [top, left, bottom, right]
            arr_shoes_oid.append(footwear)

    # ------ Bottle COCO
    for detection2 in detections_coco:
        classID = int(detection2[1])
        classe = classes_coco[classID]
        if classe == "cup" or classe == "wine glass" or classe == "bottle":
            left = int(detection2[3] * width)
            top = int(detection2[4] * height)
            right = int(detection2[5] * width)
            bottom = int(detection2[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0

            bottle = [top, left, bottom, right]
            arr_bottles_coco.append(bottle)

    ###########################################

    bPersonShoesOn = False
    bPersonDrinkOn = False
    footwearCounter = 0
    indexHuman = 0
    footwearCounter = 0

    # ------ Bottle/Drink from COCO
    for human in arr_persons_coco:
        person = human
        indexHuman = indexHuman + 1
        right_person = person[3]

        print("Human." + str(indexHuman))

        for bottle in arr_bottles_oid:
            cup = bottle
            if intersecting(person, cup, margin_drink):
                bPersonDrinkOn = True

        for bottle in arr_bottles_coco:
            cup = bottle
            if intersecting(person, cup, margin_drink):
                bPersonDrinkOn = True

        for shoe in arr_shoes_oid:
            footwear = shoe
            if intersecting(person, footwear, margin_shoes):
                footwearCounter = footwearCounter + 1

        if footwearCounter == 2:
            bPersonShoesOn = True

        text = (
            "Person."
            + str(indexHuman)
            + ".Drink."
            + str(bPersonDrinkOn)
            + ".ShoesOn."
            + str(bPersonShoesOn)
        )
        print(text)
        dictionnaire[text] = int(right_person)

    # ------ Bottle/Drink from OID
    for human in arr_persons_oid:
        person = human
        indexHuman = indexHuman + 1
        right_person = person[3]

        print("Human." + str(indexHuman))

        for bottle in arr_bottles_oid:
            cup = bottle
            if intersecting(person, cup, margin_drink):
                bPersonDrinkOn = True

        for bottle in arr_bottles_coco:
            cup = bottle
            if intersecting(person, cup, margin_drink):
                bPersonDrinkOn = True

        for shoe in arr_shoes_oid:
            footwear = shoe
            if intersecting(person, footwear, margin_shoes):
                footwearCounter = footwearCounter + 1

        if footwearCounter == 2:
            bPersonShoesOn = True

        text = (
            "Person."
            + str(indexHuman)
            + ".Drink."
            + str(bPersonDrinkOn)
            + ".ShoesOn."
            + str(bPersonShoesOn)
        )
        print(text)
        dictionnaire[text] = int(right_person)

    if len(dictionnaire) != 0:
        sorted_tuples = sorted(dictionnaire.items(), key=operator.itemgetter(1))
        for k, v in sorted_tuples:
            result_dictionnaire[k] = v

    return result_dictionnaire


def has_chairs_oid(classes, detections, imageWidth, imageHeight):
    width = imageWidth
    height = imageHeight

    arr_empty_chairs = []
    arr_taken_chairs = []
    arr_persons = []

    # ------ Human
    for detection in detections:
        classID = int(detection[1])
        classe = classes[classID - 1]
        if (
            classe == "Human face"
            or classe == "Human body"
            or classe == "Man"
            or classe == "Woman"
            or classe == "Person"
            or classe == "Boy"
            or classe == "Girl"
            or classe == "Human head"
        ):
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

    # ------ Chair
    for detection in detections:
        classID = int(detection[1])
        classe = classes[classID - 1]
        if (
            classe == "Chair"
            or classe == "Sofa bed"
            or classe == "Studio couch"
            or classe == "Couch"
            or classe == "Loveseat"
            or classe == "Stool"
        ):
            chair = Chair()
            # detection[len(detection) - 1][1] = classe
            left = int(detection[3] * width)
            top = int(detection[4] * height)
            right = int(detection[5] * width)
            bottom = int(detection[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            chair.xyxy = [top, left, bottom, right]

            margin = 20
            for person in arr_persons:
                if intersecting(person, chair.xyxy, margin):
                    chair.isEmpty = False
                    break

            if chair.isEmpty:
                arr_empty_chairs.append(chair)  # Chair instances
            else:
                arr_taken_chairs.append(chair)  # Chair instances

    print("     -->  person detected : " + str(len(arr_persons)))
    print("     -->  chair/cought FREE detected : " + str(len(arr_empty_chairs)))
    print("     -->  chair/cought TAKEN detected : " + str(len(arr_taken_chairs)))

    return arr_persons, arr_empty_chairs, arr_taken_chairs


def has_chairs_coco_couch(classes, detections, imageWidth, imageHeight):
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

    return arr_persons, arr_empty_chairs, arr_taken_chairs


def has_chairs_coco(classes, detections, imageWidth, imageHeight):
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
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            chair.xyxy = [top, left, bottom, right]

            margin = 20
            for person in arr_persons:
                if intersecting(person, chair.xyxy, margin):
                    chair.isEmpty = False
                    break

            if chair.isEmpty:
                arr_empty_chairs.append(chair)  # Chair instances
            else:
                arr_taken_chairs.append(chair)  # Chair instances

    print("     -->  person detected : " + str(len(arr_persons)))
    print("     -->  chair/cought FREE detected : " + str(len(arr_empty_chairs)))
    print("     -->  chair/cought TAKEN detected : " + str(len(arr_taken_chairs)))

    return arr_persons, arr_empty_chairs, arr_taken_chairs


def clean_coco_oid( classes_coco, detections_coco, classes_oid, detections_oid, imageWidth, imageHeight):
    print("                cleaning in progress COCO ....")

    width = imageWidth
    height = imageHeight

    tmpDetections_coco = []

    # ------ Man / Woman
    for detection_coco in detections_coco:
        bIntersection = 0
        classID_coco = int(detection_coco[1])
        classe_coco = classes_coco[classID_coco]
        score_coco = float(detection_coco[2])
        if classe_coco == "person" and score_coco > 0.3:
            left = int(detection_coco[3] * width)
            top = int(detection_coco[4] * height)
            right = int(detection_coco[5] * width)
            bottom = int(detection_coco[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            xyxy_coco = [top, left, bottom, right]

            for detection_oid in detections_oid:
                classID_oid = int(detection_oid[1])
                classe_oid = classes_oid[classID_oid - 1]
                score_oid = float(detection_oid[2])
                if (classe_oid == "Man" or classe_oid == "Woman") and score_oid > 0.3:
                    left = int(detection_oid[3] * width)
                    top = int(detection_oid[4] * height)
                    right = int(detection_oid[5] * width)
                    bottom = int(detection_oid[6] * height)
                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0
                    xyxy_oid = [top, left, bottom, right]

                    margin = 5
                    if intersecting(xyxy_coco, xyxy_oid, margin) == 1:
                        bIntersection = 1

        if bIntersection == 0 and score_coco > 0.3:
            tmpDetections_coco.append(detection_coco)
        elif score_coco > 0.3:
            print("                ---> cleaning", classe_coco)

    return tmpDetections_coco


def clean_oid_coco(
    classes_coco, detections_coco, classes_oid, detections_oid, imageWidth, imageHeight
):
    print("                cleaning in progress OID ....")

    width = imageWidth
    height = imageHeight

    finalDetections_oid = []
    tmpDetections_oid = []

    # ------ Human

    for detection_oid in detections_oid:
        bIntersection = 0
        classID_oid = int(detection_oid[1])
        classe_oid = classes_oid[classID_oid - 1]
        score_oid = float(detection_oid[2])
        if classe_oid == "Person" and score_oid > 0.3:
            left = int(detection_oid[3] * width)
            top = int(detection_oid[4] * height)
            right = int(detection_oid[5] * width)
            bottom = int(detection_oid[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            xyxy_oid = [top, left, bottom, right]

            for detection_coco in detections_coco:
                classID_coco = int(detection_coco[1])
                classe_coco = classes_coco[classID_coco]
                if classe_coco == "person":
                    left = int(detection_coco[3] * width)
                    top = int(detection_coco[4] * height)
                    right = int(detection_coco[5] * width)
                    bottom = int(detection_coco[6] * height)
                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0
                    xyxy_coco = [top, left, bottom, right]

                    margin = 5
                    if intersecting(xyxy_coco, xyxy_oid, margin) == 1:
                        bIntersection = 1

        if bIntersection == 0 and score_oid > 0.3:
            tmpDetections_oid.append(detection_oid)
        elif score_oid > 0.3:
            print("                ---> cleaning", classe_oid)

    # ------ CHAIR

    for detection_oid in tmpDetections_oid:
        bIntersection = 0
        classID_oid = int(detection_oid[1])
        classe_oid = classes_oid[classID_oid - 1]
        score_oid = float(detection_oid[2])
        if classe_oid == "Chair" and score_oid > 0.3:
            left = int(detection_oid[3] * width)
            top = int(detection_oid[4] * height)
            right = int(detection_oid[5] * width)
            bottom = int(detection_oid[6] * height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            xyxy_oid = [top, left, bottom, right]

            for detection_coco in detections_coco:
                classID_coco = int(detection_coco[1])
                classe_coco = classes_coco[classID_coco]
                if classe_coco == "chair":
                    left = int(detection_coco[3] * width)
                    top = int(detection_coco[4] * height)
                    right = int(detection_coco[5] * width)
                    bottom = int(detection_coco[6] * height)
                    if left < 0:
                        left = 0
                    if top < 0:
                        top = 0
                    xyxy_coco = [top, left, bottom, right]

                    margin = 5
                    if intersecting(xyxy_coco, xyxy_oid, margin) == 1:
                        bIntersection = 1

        if bIntersection == 0 and score_oid > 0.3:
            finalDetections_oid.append(detection_oid)
        elif score_oid > 0.3:
            print("                ---> cleaning", classe_oid)

    return finalDetections_oid


def computeDetections(self, cv_rgb, bCOCO, bOID):
    t_start_computing = time.time()
    outputs = None
    outputs_coco = None

    # detect object
    if bOID == 1:
        blob = cv2.dnn.blobFromImage(cv_rgb, size=(
            300, 300), swapRB=False, crop=False)
        self.cvNet.setInput(blob)
        outputs = self.cvNet.forward()

        # ------------------ TIMING  ------------------------------
        t_end_computing = time.time()
        if DISPLAY_ALL_DELAY == 1:
            print(bcolors.B+"     --> computeDetections (OID) delay " +
                    str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)
            t_start_computing = time.time()

    if bCOCO == 1:
        blob2 = cv2.dnn.blobFromImage(
            cv_rgb, size=(300, 300), swapRB=True, crop=False)
        self.cvNet_coco.setInput(blob2)
        outputs_coco = self.cvNet_coco.forward()
        # ------------------ TIMING  ------------------------------
        t_end_computing = time.time()
        if DISPLAY_ALL_DELAY == 1:
            print(bcolors.B+"     --> computeDetections (COCO) delay " +
                    str(round(t_end_computing - t_start_computing, 3))+bcolors.ENDC)

    return outputs, outputs_coco