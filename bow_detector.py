################################################################################

# functionality: perform detection based on Bag of (visual) Words SVM classification
# using a very basic multi-scale, sliding window (exhaustive search) approach

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Origin ackowledgements: forked from https://github.com/siphomateke/PyBOW

################################################################################

import cv2
import os
import numpy as np
import math
import params
from utils import *
from sliding_window import *

################################################################################

directory_to_cycle = "pedestrian/INRIAPerson/Test/pos/";

show_scan_window_process = True;

################################################################################

# load dictionary and SVM data

try:
    dictionary = np.load(params.BOW_DICT_PATH)
    svm = cv2.ml.SVM_load(params.BOW_SVM_PATH)
except:
    print("Missing files - dictionary and/or SVM!");
    print("-- have you performed training to produce these files ?");
    exit();

# print some checks

print("dictionary size : ", dictionary.shape)
print("svm size : ", len(svm.getSupportVectors()))
print("svm var count : ", svm.getVarCount())

################################################################################

# process all images in directory (sorted by filename)

for filename in sorted(os.listdir(directory_to_cycle)):

    # if it is a PNG file

    if '.png' in filename:
        print(os.path.join(directory_to_cycle, filename));

        # read image data

        img = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)

        # make a copy for drawing the output

        output_img = img.copy();

        # for a range of different image scales in an image pyramid

        current_scale = -1
        detections = []
        rescaling_factor = 1.25

        ################################ for each re-scale of the image

        for resized in pyramid(img, scale=rescaling_factor):

            # at the start our scale = 1, because we catch the flag value -1

            if current_scale == -1:
                current_scale = 1

            # after this rescale downwards each time (division by re-scale factor)

            else:
                current_scale /= rescaling_factor

            rect_img = resized.copy()

            # if we want to see progress show each scale

            if (show_scan_window_process):
                cv2.imshow('current scale',rect_img)
                cv2.waitKey(10);

            # loop over the sliding window for each layer of the pyramid (re-sized image)

            window_size = params.DATA_WINDOW_SIZE
            step = math.floor(resized.shape[0] / 16)

            if step > 0:

                ############################# for each scan window

                for (x, y, window) in sliding_window(resized, window_size, step_size=step):

                    # if we want to see progress show each scan window

                    if (show_scan_window_process):
                        cv2.imshow('current window',window)
                        key = cv2.waitKey(10) # wait 10ms

                    # for each window region get the BoW feature point descriptors

                    img_data = ImageData(window)
                    img_data.compute_bow_descriptors()

                    # generate and classify each window by constructing a BoW
                    # histogram and passing it through the SVM classifier

                    if img_data.bow_descriptors is not None:
                        img_data.generate_bow_hist(dictionary)

                        print("detecting with SVM ...")

                        retval, [result] = svm.predict(np.float32([img_data.bow_histogram]))

                        print(result)

                        # if we get a detection, then record it

                        if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:

                            # store rect as (x1, y1) (x2,y2) pair

                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])

                            # if we want to see progress show each detection, at each scale

                            if (show_scan_window_process):
                                cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                                cv2.imshow('current scale',rect_img)
                                cv2.waitKey(10)

                            rect *= (1.0 / current_scale)
                            detections.append(rect)

                ########################################################

        # For the overall set of detections (over all scales) perform
        # non maximal suppression (i.e. remove overlapping boxes etc).

        detections = non_max_suppression_fast(np.int32(detections), 0.4)

        # finally draw all the detection on the original image

        for rect in detections:
            cv2.rectangle(output_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

        cv2.imshow('detected objects',output_img)
        key = cv2.waitKey(40) # wait 40ms
        if (key == ord('x')):
            break

# close all windows

cv2.destroyAllWindows()

#####################################################################
