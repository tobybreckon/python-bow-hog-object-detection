################################################################################

# functionality: perform Bag of (visual) Words (BoW) testing over
# a specified dataset and compute the resulting prediction/clasification error
# over that same dataset, using pre-saved the SVM model + BOW dictionary

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Origin acknowledgements: forked from https://github.com/siphomateke/PyBOW

################################################################################

import numpy as np
import cv2
from utils import *

################################################################################

def main():

    # load up the dictionary and SVM stored from prior training

    try:
        dictionary = np.load(params.BOW_DICT_PATH)
        svm = cv2.ml.SVM_load(params.BOW_SVM_PATH)
    except:
        print("Missing files - dictionary and/or SVM!");
        print("-- have you performed training to produce these files ?");
        exit();

    # load ** testing ** data sets in the same class order as training
    # (here we perform no patch sampling of the data as we are not training
    # hence [0,0] sample sizes and [False,False] centre weighting flags)

    print("Loading test data as a batch ...")

    paths = [params.DATA_testing_path_neg, params.DATA_testing_path_pos]
    use_centre_weighting = [False, False];
    class_names = params.DATA_CLASS_NAMES
    imgs_data = load_images(paths, class_names, [0,0], use_centre_weighting)

    print("Computing descriptors...") # for each testing image
    start = cv2.getTickCount()
    [img_data.compute_bow_descriptors() for img_data in imgs_data]
    print_duration(start)

    print("Generating histograms...") # for each testing image
    start = cv2.getTickCount()
    [img_data.generate_bow_hist(dictionary) for img_data in imgs_data]
    print_duration(start)

    # get the example/sample bow histograms and class labels

    samples, class_labels = get_bow_histograms(imgs_data), get_class_labels(imgs_data)

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
