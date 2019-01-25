################################################################################

# functionality: parameter settings for detection algorithm training/testing

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Origin acknowledgements: forked from https://github.com/siphomateke/PyBOW

################################################################################

import cv2
import os

################################################################################
# settings for datsets in general

master_path_to_dataset = "/tmp/pedestrian"; # ** need to edit this **

# data location - training examples

DATA_training_path_neg = os.path.join(master_path_to_dataset,"INRIAPerson/Train/neg/");
DATA_training_path_pos = os.path.join(master_path_to_dataset,"INRIAPerson/train_64x128_H96/pos/");

# data location - testing examples

DATA_testing_path_neg = os.path.join(master_path_to_dataset,"INRIAPerson/Test/neg/");
DATA_testing_path_pos = os.path.join(master_path_to_dataset,"INRIAPerson/test_64x128_H96/pos/");

# size of the sliding window patch / image patch to be used for classification
# (for larger windows sizes, for example from selective search - resize the
# window to this size before feature descriptor extraction / classification)

DATA_WINDOW_SIZE = [64, 128];

# the maximum left/right, up/down offset to use when generating samples for training
# that are centred around the centre of the image

DATA_WINDOW_OFFSET_FOR_TRAINING_SAMPLES = 3;

# number of sample patches to extract from each negative training example

DATA_training_sample_count_neg = 10;

# number of sample patches to extract from each positive training example

DATA_training_sample_count_pos = 5;

# class names - N.B. ordering of 0, 1 for neg/pos = order of paths

DATA_CLASS_NAMES = {
    "other": 0,
    "pedestrian": 1
}

################################################################################
# settings for BOW - Bag of (visual) Word - approaches

BOW_SVM_PATH = "svm_bow.xml"
BOW_DICT_PATH = "bow_dictionary.npy"

BOW_dictionary_size = 512;  # in general, larger = better performance, but potentially slower
BOW_SVM_kernel = cv2.ml.SVM_RBF; # see opencv manual for other options
BOW_SVM_max_training_iterations = 500; # stop training after max iterations

BOW_clustering_iterations = 20; # reduce to improve speed, reduce quality

BOW_fixed_feature_per_image_to_use = 100; # reduce to improve speed, set to 0 for variable number

# specify the type of feature points to use])
# -- refer to the OpenCV manual for options here, by default this is set to work on
# --- all systems "out of the box" rather than using the best available option

BOW_use_ORB_always = False; # set to True to always use ORB over SIFT where available

try:

    if BOW_use_ORB_always:
        print("Forced used of ORB features, not SIFT")
        raise Exception('force use of ORB')

    DETECTOR = cv2.xfeatures2d.SIFT_create(nfeatures=BOW_fixed_feature_per_image_to_use) # -- requires extra modules and non-free build flag
    # DETECTOR = cv2.xfeatures2d.SURF_create(nfeatures=BOW_fixed_feature_per_image_to_use) # -- requires extra modules and non-free build flag

    # as SIFT/SURF feature descriptors are floating point use KD_TREE approach

    _algorithm = 0 # FLANN_INDEX_KDTREE
    _index_params = dict(algorithm=_algorithm, trees=5)
    _search_params = dict(checks=50)

except:

    DETECTOR = cv2.ORB_create(nfeatures=BOW_fixed_feature_per_image_to_use) # check these params

    #if using ORB points
    # taken from: https://docs.opencv.org/3.3.0/dc/dc3/tutorial_py_matcher.html
    # N.B. "commented values are recommended as per the docs,
    # but it didn't provide required results in some cases"

    # as SIFT/SURF feature descriptors are integer use HASHING approach

    _algorithm = 6 # FLANN_INDEX_LSH
    _index_params= dict(algorithm = _algorithm,
                        table_number = 6, # 12
                        key_size = 12,     # 20
                        multi_probe_level = 1) #2
    _search_params = dict(checks=50)

    if (not(BOW_use_ORB_always)):
        print("Falling back to using features: ", DETECTOR.__class__())
        BOW_use_ORB_always = True; # set this as a flag we can check later which data type to uses

print("For BOW - features in use are: ", DETECTOR.__class__(), "(ignore for HOG)")

# based on choice and availability of feature points, set up KD-tree matcher

MATCHER = cv2.FlannBasedMatcher(_index_params, _search_params)

################################################################################
# settings for HOG approaches

HOG_SVM_PATH = "svm_hog.xml"

HOG_SVM_kernel = cv2.ml.SVM_LINEAR; # see opencv manual for other options
HOG_SVM_max_training_iterations = 500; # stop training after max iterations

################################################################################
