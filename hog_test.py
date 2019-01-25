################################################################################

# functionality: perform HOG/SVM testing over a specified dataset and compute the
# resulting prediction/clasification error over that same dataset, using
# pre-saved the SVM model trained on HOG feature descriptors

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Minor portions: based on fork from https://github.com/siphomateke/PyBOW

################################################################################

import numpy as np
import cv2
from utils import *

################################################################################

def main():

    # load up the SVM stored from prior training

    try:
        svm = cv2.ml.SVM_load(params.HOG_SVM_PATH)
    except:
        print("Missing files  SVM");
        print("-- have you performed training to produce this file ?");
        exit();

    # load ** testing ** data sets in the same class order as training
    # (here we perform patch sampling only from the centre of the +ve
    # class and only a single sample is taken
    # hence [0,0] sample sizes and [False,True] centre weighting flags)

    print("Loading test data as a batch ...")

    paths = [params.DATA_testing_path_neg, params.DATA_testing_path_pos]
    use_centre_weighting = [False, True];
    class_names = params.DATA_CLASS_NAMES
    imgs_data = load_images(paths, class_names, [0,0], use_centre_weighting)

    print("Computing HOG descriptors...") # for each testing image
    start = cv2.getTickCount()
    [img_data.compute_hog_descriptor() for img_data in imgs_data]
    print_duration(start)

    # get the example/sample HOG descriptors and class labels

    samples, class_labels = get_hog_descriptors(imgs_data), get_class_labels(imgs_data)

    # perform batch SVM classification over the whole set

    print("Performing batch SVM classification over all data  ...")

    results = svm.predict(samples)
    output = results[1].ravel()

    # compute and report the error over the whole set

    error = ((np.absolute(class_labels.ravel() - output).sum()) / float(output.shape[0]))
    print("Successfully trained SVM with {}% testing set error".format(round(error * 100,2)))
    print("-- meaining the SVM got {}% of the testing examples correct!".format(round((1.0 - error) * 100,2)))

################################################################################

if __name__ == "__main__":
    main()

################################################################################
