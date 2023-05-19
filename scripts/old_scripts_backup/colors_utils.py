import numpy as np
import cv2
from datetime import datetime
import os

from perception_utils.utils import get_pkg_path
# from perception_utils.get_color_name_python3 import csv_reader, get_color_name, get_simple_color_names
pkg_path = get_pkg_path()
newest_csv_path = os.path.join(pkg_path, 'scripts/res/new_colorsV2.csv')
brightness_value = 10


# newest_csv_path = '../res/new_colors.csv'
#########################################################################################
# TERMINAL color
#########################################################################################
W = '\033[0m'  # white (normal)
R = '\033[31m'  # red
G = '\033[32m'  # green
O = '\033[33m'  # orange
B = '\033[34m'  # blue
P = '\033[35m'  # purple


def change_brightness(img, brightness_value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,brightness_value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

#########################################################################################
def detect_colors(image, num_clusters, num_iters, resize_factor, crop_factor, colors_in_order, csv, type="rgb", name=""):
    # start_time = datetime.now()

    if image.size == 0:
        print(O + '     detect_color image original EMPTY' + W)
        return 0, 0, 0, None

    image = change_brightness(image, brightness_value)
    # crop
    height, width, depth = image.shape
    crop_factor = (100 - crop_factor) / 2
    # print(crop_factor)
    image = image[int(height * crop_factor / 100):(height - int(height * crop_factor / 100)),
                  int(width * crop_factor / 100):(width - int(width * crop_factor / 100))]
    
    if image.size == 0:
        print(O + '     detect_color image cropped EMPTY' + W)
        return 0, 0, 0, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # print("shape:" , image.shape)
    w_resize = int(image.shape[1] * resize_factor / 100)  # vertical
    h_resize = int(image.shape[0] * resize_factor / 100)  # horizontal
    # print(w_resize,h_resize)
    if h_resize == 0 or w_resize == 0:
        #print(O + '     detect_color h_resize or w_resize = 0 set to 1' + W)
        w_resize = 1
        h_resize = 1

    dim = (w_resize, h_resize)

    # resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
    w, h, d = image.shape
    image_flat = np.reshape(image, (w * h, 3))
    # print(image)
    # cv2.imwrite("result.png", image)
    Z = image_flat
    Z = np.float32(Z)
    if Z.shape[0] <= num_clusters:
        num_clusters = Z.shape #-1
    # if Z.shape[0] <= num_clusters:
    #     print(O + '     KMEANS failed -- > Z.shape[0] < = num_clusters : ' + str(Z.shape[0]) + ' < = ' + str(num_clusters) + W)
    #     return 0, 0, 0, None

    if (type == "hue"):
        # convert to HSV and use Hue value only
        # resize image
        # print(image_hsv)
        image_hsv = cv2.resize(image_hsv, dim, interpolation=cv2.INTER_NEAREST)
        # print(image_hsv)
        w, h, d = image_hsv.shape
        image_flat_hsv = np.reshape(image_hsv, (w * h, 3))
        # print(w,h,d)
        Z_hsv = image_flat_hsv
        Z_hsv = np.float32(Z_hsv)
        Z_hsv = Z_hsv[:, 0]
        # print(Z_hsv)
        # define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, num_iters, 1.0)
        ret, label, center = cv2.kmeans(
            Z_hsv, num_clusters, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)


        # find centers by RGB values of cluster samplescolour_csv
        percentages = []
        centers = []
        for i in range(num_clusters):
            samples = Z[label.ravel() == i]  # RGB values
            center = np.mean(samples, axis=0)
            centers.append(center)

            # mapping the clusters
            for j in range(len(Z)):
                if (label[j] == i):
                    Z[j] = center

            percentage = len(samples) * 100 / Z.shape[0]
            percentages.append(percentage)
            percentage_str = str(i) + "-" + str(percentage) + "%"

        # print("centers", center.dtype)
        # print("percentages",percentage.dtype)
        centers_sorted = sort_color_by_percentage(centers, percentages)
        # delay = datetime.now() - start_time
        # print("**************** delay", delay.total_seconds())

        # reconstruct image
        Z = Z.reshape((w, h, 3))
        # print(centers_sorted)
        percentages.sort()

        for x in range(len(centers_sorted)):
            max_color = centers_sorted[x]
            # print(closest(
            #     max_color[0], max_color[1], max_color[2]),colors_in_order,csv)
            # print(percentages[x])

        return centers_sorted[len(centers_sorted) - 1], percentages[len(percentages) - 1]

    elif (type == "rgb"):
        # define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, num_iters, 1.0)
        ret, label, center = cv2.kmeans( Z, num_clusters, None, criteria, 100, cv2.KMEANS_PP_CENTERS)  # cv2.KMEANS_RANDOM_CENTERS)

        percentages = []
        for i in range(num_clusters):
            samples = Z[label.ravel() == i]

            # mapping the clusters
            for j in range(len(Z)):
                if (label[j] == i):
                    Z[j] = center[i]

            percentage = len(samples) * 100 / Z.shape[0]
            percentages.append(percentage)
            # percentage_str = str(i) + "-" + str(percentage) + "%"
        # print(percentages)
        # TODO: return mapping
        Z = Z.reshape((w, h, 3))
        mapping = cv2.cvtColor(Z, cv2.COLOR_RGB2BGR)
        mapping = np.uint8(mapping)
        # cv2.imshow('mapping_rgb', cv2.cvtColor(Z, cv2.COLOR_RGB2BGR))

        # FIXME: save mapping picture every 10 seconds
        # date_time = datetime.now()
        # if (int(date_time.split("_")[5]) % 10 == 0):
        #     if (not os.path.exists('output/')):
        #         os.mkdir('output/')
        #     cv2.imwrite('output/mapping_rgb_' + name + '_' + date_time + '.jpg', cv2.cvtColor(Z, cv2.COLOR_RGB2BGR))

        # percentages.sort()

        max_idx = np.argmax(percentages)
        max_color = center[max_idx]
        color_res = closest(
            max_color[0], max_color[1], max_color[2],colors_in_order,csv)
        return 1, color_res, max_color, mapping

    else:
        print("Error: type not defined. Type must be either hue or rgb.")
        return 0, None, None, None


#########################################################################################
def detect_colors_without_mapping(image, num_clusters, num_iters, resize_factor, crop_factor, colors_in_order, csv, type="rgb"):
    # start_time = datetime.now()

    if image.size == 0:
        print(O + '     detect_color image original EMPTY' + W)
        return 0, 0, 0

    image = change_brightness(image, brightness_value)
    # crop
    height, width, depth = image.shape
    crop_factor = (100 - crop_factor) / 2

    image = image[int(height * crop_factor / 100):(height - int(height * crop_factor / 100)),
                  int(width * crop_factor / 100):(width - int(width * crop_factor / 100))]

    if image.size == 0:
        print(O + '     detect_color image cropped EMPTY' + W)
        return 0, 0, 0, None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    w_resize = int(image.shape[1] * resize_factor / 100)  # vertical
    h_resize = int(image.shape[0] * resize_factor / 100)  # horizontal

    if h_resize == 0 or w_resize == 0:
        #print(O + '     detect_color h_resize or w_resize = 0 set to 1' + W)
        w_resize = 1
        h_resize = 1

    dim = (w_resize, h_resize)

    # resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_NEAREST)
    w, h, d = image.shape
    image_flat = np.reshape(image, (w * h, 3))

    Z = image_flat
    Z = np.float32(Z)

    if Z.shape[0] <= num_clusters:
        #print(O + '     KMEANS failed -- > Z.shape[0] < = num_clusters : ' + str(Z.shape[0]) + ' < = ' + str(num_clusters) + W)
        num_clusters = Z.shape[0] - 1
        # return 0, 0, 0, None

    if (type == "hue"):
        # convert to HSV and use Hue value only
        # resize image
        image_hsv = cv2.resize(image_hsv, dim, interpolation=cv2.INTER_NEAREST)
        w, h, d = image_hsv.shape
        image_flat_hsv = np.reshape(image_hsv, (w * h, 3))

        Z_hsv = image_flat_hsv
        Z_hsv = np.float32(Z_hsv)
        Z_hsv = Z_hsv[:, 0]

        # define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, num_iters, 1.0)
        ret, label, center = cv2.kmeans(
            Z_hsv, num_clusters, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

        # find centers by RGB values of cluster samples
        percentages = []
        centers = []
        for i in range(num_clusters):
            samples = Z[label.ravel() == i]  # RGB values
            center = np.mean(samples, axis=0)
            centers.append(center)

            # mapping the clusters
            for j in range(len(Z)):
                if (label[j] == i):
                    Z[j] = center

            percentage = len(samples) * 100 / Z.shape[0]
            percentages.append(percentage)
            percentage_str = str(i) + "-" + str(percentage) + "%"

        # print("centers", center)
        centers_sorted = sort_color_by_percentage(centers, percentages)
        # delay = datetime.now() - start_time
        # print("**************** delay", delay.total_seconds())

        # reconstruct image
        Z = Z.reshape((w, h, 3))

        percentages.sort()
        max_color = centers_sorted[len(centers_sorted) - 1]
        # print(closest(
        #     max_color[0], max_color[1], max_color[2]),colors_in_order,csv)
        return centers_sorted[len(centers_sorted) - 1], percentages[len(percentages) - 1]

    elif (type == "rgb"):
        # define criteria and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, num_iters, 1.0)

        ret, label, center = cv2.kmeans(
            Z, num_clusters, None, criteria, 100, cv2.KMEANS_PP_CENTERS)  # cv2.KMEANS_RANDOM_CENTERS)

        percentages = []
        for i in range(num_clusters):
            samples = Z[label.ravel() == i]

            # mapping the clusters
            for j in range(len(Z)):
                if (label[j] == i):
                    Z[j] = center[i]

            percentage = len(samples) * 100 / Z.shape[0]
            percentages.append(percentage)
            # percentage_str = str(i) + "-" + str(percentage) + "%"

        max_idx = np.argmax(percentages)
        max_color = center[max_idx]
        color_res = closest(
            max_color[0], max_color[1], max_color[2],colors_in_order,csv)
        # TODO: return mapping
        # Z = Z.reshape((w, h, 3))
        # mapping = cv2.cvtColor(Z, cv2.COLOR_RGB2BGR)
        # mapping = np.uint8(mapping)
        # cv2.imshow('mapping_rgb', cv2.cvtColor(Z, cv2.COLOR_RGB2BGR))

        # FIXME: save mapping picture every 10 seconds
        # date_time = datetime.now()
        # if (int(date_time.split("_")[5]) % 10 == 0):
        #     if (not os.path.exists('output/')):
        #         os.mkdir('output/')225, 228, 255=False)
        return 1, color_res #, max_color #, mapping

    else:
        print("Error: type not defined. Type must be either hue or rgb.")
        return 0, None, None #, None
#########################################################################################
def csv_reader(datafile, has_header):
    data = []
    if has_header:
        with open(datafile, "r") as f:
            header = f.readline().split(",")
            counter = 0
            for line in f:
                data.append(line)
                fields = line.split(",")
                counter += 1
    else:
        with open(datafile, "r") as f:
            data = f.read().splitlines()
        data = [line.split(',#') for line in data]
    return data

def new_csv_reader(datafile):
    data = []
    with open(datafile, "r", encoding='utf-8-sig') as f:
        data = f.read().splitlines()
    data = [line.split(',') for line in data]
    return data

#########################################################################################

# Find the closest color in color list
# colors in order is an array contains only RGB values in order
# csv is used to find the closest color name
def closest (R,G,B, colors_in_order, csv):
    RGB = np.array([R,G,B])
    # Find Euclidean distance
    distances = np.sqrt(np.sum((colors_in_order-RGB)**2,axis=1))
    # Get the smallest distance
    index_of_smallest = np.where(distances==np.amin(distances))
    return csv[int(index_of_smallest[0][0])][1]

#########################################################################################
def sort_color_by_percentage(colors, percentages):
    return [x for _, x in sorted(zip(percentages, colors),key = lambda tup:tup[0])]

#########################################################################################
# if __name__ == "__main__":
#     cam_port = 0
#     cam = cv2.VideoCapture(cam_port)
#     result, image = cam.read()
#     # cv2.imwrite("pic.png", image)
#     # read fle new_colors.csv
    

#     data = new_csv_reader(newest_csv_path)
#     row_count = sum(1 for line in data) 
#     # print(row_count)
#     colors_in_order = np.zeros(shape=(row_count,3))
#     for i in range(len(data)):
#         ref_color = data[i]
#         # print(ref_color)
#         colors_in_order[i] = list(map(int,ref_color[0].split('-')[:3]))

#     # image = cv2.imread("pic.png", cv2.IMREAD_COLOR)
#     # image2 = cv2.createBackgroundSubtractorKNN()
#     # cv2.imwrite("removed_background.png", image2)
#     # print(image)
#     # print(rgb_to_hex(detect_colors(image, 10, 10, 10, 10, type="hue", name="")[2].astype(int)))
#     print(detect_colors(image, 2, 50, 10, 70, colors_in_order, data, type="rgb", name=""))
#     # print(detect_colors_without_mapping(image, 2, 50, 10, 100, type="rgb"))
#     # data = new_csv_reader('../res/new_colors.csv')
#     # print(colors_in_order)
#     print(closest(157,45,38,colors_in_order, data))
#     # img2_fg = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#     # cv2.imwrite("mask.png",img2_fg)
