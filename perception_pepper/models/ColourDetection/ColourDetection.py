import numpy as np
import cv2
from datetime import datetime
import os
from perception_pepper.perception_utils.utils import get_pkg_path
from perception_pepper.perception_utils.bcolors import bcolors
import time

class ColourDetection():
    
    def __init__(self, colour_csv_file_name="new_colorsV3.csv", color_type="rgb"):
        
        self.pkg_path = get_pkg_path()
        self.newest_csv_path = os.path.join(self.pkg_path, ('models/ColourDetection/color_csv/' + colour_csv_file_name))
        self.brightness_value = 5
        self.contrast = 5
        self.crop_factor = 60
        self.color_type = color_type
        print(
                bcolors.CYAN+"[RoboBreizh - Vision]    Loading Colour Detection type --" + self.color_type +  "-- done"+bcolors.ENDC)

    def change_brightness(self,img, brightness_value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v,brightness_value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img
    
    def apply_brightness_contrast(self,input_img):
        
        brightness = self.brightness_value
        contrast = self.contrast
    
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha_b = (highlight - shadow)/255
            gamma_b = shadow
            
            buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
        else:
            buf = input_img.copy()
        
        if contrast != 0:
            f = 131*(contrast + 127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            
            buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

        return buf

    def csv_reader(self, datafile, has_header):
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

    def new_csv_reader(self,datafile):
        data = []
        with open(datafile, "r", encoding='utf-8-sig') as f:
            data = f.read().splitlines()
        data = [line.split(',') for line in data]

        return data
    
    def closest (self, R,G,B, colors_in_order, csv):
        RGB = np.array([R,G,B])
        # Find Euclidean distance
        distances = np.sqrt(np.sum((colors_in_order-RGB)**2,axis=1))
        # Get the smallest distance
        index_of_smallest = np.where(distances==np.amin(distances))
        
        return csv[int(index_of_smallest[0][0])][1]
    
    def sort_color_by_percentage(self,colors, percentages):
        return [x for _, x in sorted(zip(percentages, colors),key = lambda tup:tup[0])]

    def inference(self, cropped, classe):
        
        time_start = time.time()
        
        """
        :return: ok (boolean), color_res (string) , mapping (image)
        """
        color_res = ''
        t400 = time.time()
        color_resize_factor = 10
        data = self.new_csv_reader(self.newest_csv_path)
        row_count = sum(1 for line in data)
        colors = np.zeros(shape=(row_count, 3))
        for i in range(len(data)):
            ref_color = data[i]
            colors[i] = list(map(int, ref_color[0].split('-')[:3]))

        if len(cropped) != 0:
            ok, color_res, center_sorted, mapping = self.detect_colors(
                cropped, num_clusters=3, num_iters=50, resize_factor=color_resize_factor, crop_factor=self.crop_factor, colors_in_order=colors, csv=data, 
                type=self.color_type, name=str(classe))
        else:
            return 0, '', None, []
        
        time_end = time.time()
        
        print("Colour Detection Inference time : " + str(time_end-time_start))

        return ok, color_res, None, mapping

    def detect_colors(self,image, num_clusters, num_iters, resize_factor, crop_factor, colors_in_order, csv, type="rgb", name=""):

        if image.size == 0:
            print(bcolors.R + '     detect_color image original EMPTY' + bcolors.WARNING)
            return 0, 0, 0, None

        image = self.apply_brightness_contrast(image)
        # crop
        height, width, depth = image.shape
        crop_factor = (100 - crop_factor) / 2
        image = image[int(height * crop_factor / 100):(height - int(height * crop_factor / 100)),
                    int(width * crop_factor / 100):(width - int(width * crop_factor / 100))]
        
        if image.size == 0:
            print(bcolors.R + '     detect_color image cropped EMPTY' + bcolors.WARNING)
            return 0, 0, 0, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        w_resize = int(image.shape[1] * resize_factor / 100)  # vertical
        h_resize = int(image.shape[0] * resize_factor / 100)  # horizontal
        if h_resize == 0 or w_resize == 0:
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
            num_clusters = Z.shape[0] #-1

        
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


            centers_sorted = self.sort_color_by_percentage(centers, percentages)

            # reconstruct image
            Z = Z.reshape((w, h, 3))
            
            if Z.shape[0] <= num_clusters:
                num_clusters = Z.shape[0] 
            
            percentages.sort()

            for x in range(len(centers_sorted)):
                max_color = centers_sorted[x]

            color_array = centers_sorted[len(centers_sorted) - 1], percentages[len(percentages) - 1]
                        
            color_res = self.closest(
                    color_array[0][0], color_array[0][1], color_array[0][2], colors_in_order,csv)  
                        
            return 1, color_res, 0 , []

        elif (type == "rgb"):
            # define criteria and apply kmeans()
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, num_iters, 1.0)
            
            ret, label, center = cv2.kmeans( Z, num_clusters, None, criteria, 100, cv2.KMEANS_PP_CENTERS)  # cv2.KMEANS_RANDOM_CENTERS)

            percentages = []
            
            if num_clusters > 1:
                for i in range(num_clusters):
                    samples = Z[label.ravel() == i]

                    # mapping the clusters
                    for j in range(len(Z)):
                        if (label[j] == i):
                            Z[j] = center[i]

                    percentage = len(samples) * 100 / Z.shape[0]
                    percentages.append(percentage)
                # TODO: return mapping
                Z = Z.reshape((w, h, 3))
                mapping = cv2.cvtColor(Z, cv2.COLOR_RGB2BGR)
                mapping = np.uint8(mapping)

                max_idx = np.argmax(percentages)
                max_color = center[max_idx]
                color_res = self.closest(
                    max_color[0], max_color[1], max_color[2],colors_in_order,csv)
                
                return 1, color_res, max_color, mapping
            else:
                color_res = self.closest(
                    Z[0][0], Z[0][1], Z[0][2], colors_in_order,csv)             
                   
                return 1, color_res, 0, []

        else:
            print("Error: type not defined. Type must be either hue or rgb.")
            return 0, None, None, None

    def detect_colors_without_mapping(self,image, num_clusters, num_iters, resize_factor, crop_factor, colors_in_order, csv, type="rgb"):

        if image.size == 0:
            print(bcolors.O + '     detect_color image original EMPTY' + bcolors.WARNING)
            return 0, 0, 0

        image = self.apply_brightness_contrast(image)
        # crop
        height, width, depth = image.shape
        crop_factor = (100 - crop_factor) / 2

        image = image[int(height * crop_factor / 100):(height - int(height * crop_factor / 100)),
                    int(width * crop_factor / 100):(width - int(width * crop_factor / 100))]

        if image.size == 0:
            print(bcolors.O + '     detect_color image cropped EMPTY' + bcolors.WARNING)
            return 0, 0, 0, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        w_resize = int(image.shape[1] * resize_factor / 100)  # vertical
        h_resize = int(image.shape[0] * resize_factor / 100)  # horizontal

        if h_resize == 0 or w_resize == 0:
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
            num_clusters = Z.shape[0] - 1

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

            centers_sorted = self.sort_color_by_percentage(centers, percentages)

            # reconstruct image
            Z = Z.reshape((w, h, 3))

            percentages.sort()
            max_color = centers_sorted[len(centers_sorted) - 1]

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

            max_idx = np.argmax(percentages)
            max_color = center[max_idx]
            color_res = self.closest(
                max_color[0], max_color[1], max_color[2],colors_in_order,csv)

            return 1, color_res 

        else:
            print("Error: type not defined. Type must be either hue or rgb.")
            return 0, None, None #, None
        
 