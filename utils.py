################################################################################

# functionality: utility functions for BOW and HOG detection algorithms

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Origin acknowledgements: forked from https://github.com/siphomateke/PyBOW

################################################################################

import os
import numpy as np
import cv2
import params
import math
import random

################################################################################
# global flags to facilitate output of additional info per stage/function

show_additional_process_information = False;
show_images_as_they_are_loaded = False;
show_images_as_they_are_sampled = False;

################################################################################

# timing information - for training
# - helper function for timing code execution

def get_elapsed_time(start):
    return (cv2.getTickCount() - start) / cv2.getTickFrequency()


def format_time(time):
    time_str = ""
    if time < 60.0:
        time_str = "{}s".format(round(time, 1))
    elif time > 60.0:
        minutes = time / 60.0
        time_str = "{}m : {}s".format(int(minutes), round(time % 60, 2))
    return time_str


def print_duration(start):
    time = get_elapsed_time(start)
    print(("Took {}".format(format_time(time))))

################################################################################

# reads all the images in a given folder path and returns the results

# for obvious reasons this will break with a very large dataset as you will run
# out of memory - so an alternative approach may be required in that case

def read_all_images(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    for image_path in images_path:

        # add in a check to skip non jpg or png (lower case) named files
        # as some OS (Mac OS!) helpfully creates a Thumbs.db or similar
        # when you browse image folders - which then are not images when
        # we try to load them

        if (('.png' in image_path) or ('.jpg' in image_path)):
            img = cv2.imread(image_path)
            images.append(img)
            if show_additional_process_information:
                print("loading file - ", image_path);
        else:
            if show_additional_process_information:
                print("skipping non PNG/JPG file - ", image_path);

    return images

################################################################################

# stack array of items as basic Pyton data manipulation

def stack_array(arr):
    stacked_arr = np.array([])
    for item in arr:
        # Only stack if it is not empty
        if len(item) > 0:
            if len(stacked_arr) == 0:
                stacked_arr = np.array(item)
            else:
                stacked_arr = np.vstack((stacked_arr, item))
    return stacked_arr

################################################################################

# transform between class numbers (i.e. codes) - {0,1,2, ...N} and
# names {dog,cat cow, ...} - used in training and testing

def get_class_number(class_name):
    return params.DATA_CLASS_NAMES.get(class_name, 0)

def get_class_name(class_code):
    for name, code in params.DATA_CLASS_NAMES.items():
        if code == class_code:
            return name

################################################################################

# image data class object that contains the images, descriptors and bag of word
# histograms

class ImageData(object):
    def __init__(self, img):
        self.img = img
        self.class_name = ""
        self.class_number = None

        # use default parameters for construction of HOG
        # examples of non-default parameter use here:
        # https://www.programcreek.com/python/example/84776/cv2.HOGDescriptor

        self.hog = cv2.HOGDescriptor(); # default is 64 x 128
        self.hog_descriptor = np.array([])
        self.bow_descriptors = np.array([])


    def set_class(self, class_name):
        self.class_name = class_name
        self.class_number = get_class_number(self.class_name)
        if show_additional_process_information:
            print("class name : ", class_name, " - ", self.class_number);

    def compute_hog_descriptor(self):

        # generate the HOG descriptors for a given image

        img_hog = cv2.resize(self.img, (params.DATA_WINDOW_SIZE[0], params.DATA_WINDOW_SIZE[1]), interpolation = cv2.INTER_AREA)

        self.hog_descriptor = self.hog.compute(img_hog)

        if self.hog_descriptor is None:
            self.hog_descriptor = np.array([])

        if show_additional_process_information:
            print("HOG descriptor computed - dimension: ", self.hog_descriptor.shape);

    def compute_bow_descriptors(self):

        # generate the feature descriptors for a given image

        self.bow_descriptors = params.DETECTOR.detectAndCompute(self.img, None)[1]

        if self.bow_descriptors is None:
            self.bow_descriptors = np.array([])

        if show_additional_process_information:
            print("# feature descriptors computed - ", len(self.bow_descriptors));

    def generate_bow_hist(self, dictionary):
        self.bow_histogram = np.zeros((len(dictionary), 1))

        # generate the bow histogram of feature occurance from descriptors

        if (params.BOW_use_ORB_always):
            # FLANN matcher with ORB needs dictionary to be uint8
            matches = params.MATCHER.match(self.bow_descriptors, np.uint8(dictionary));
        else:
            # FLANN matcher with SIFT/SURF needs descriptors to be type32
            matches = params.MATCHER.match(np.float32(self.bow_descriptors), dictionary)

        for match in matches:
            # Get which visual word this descriptor matches in the dictionary
            # match.trainIdx is the visual_word
            # Increase count for this visual word in histogram (known as hard assignment)
            self.bow_histogram[match.trainIdx] += 1

        # Important! - normalize the histogram to L1 to remove bias for number
        # of descriptors per image or class (could use L2?)

        self.bow_histogram = cv2.normalize(self.bow_histogram, None, alpha=1, beta=0, norm_type=cv2.NORM_L1);

################################################################################

# generates a set of random sample patches from a given image of a specified size
# with an optional flag just to train from patches centred around the centre of the image

def generate_patches(img, sample_patches_to_generate=0, centre_weighted=False,
                            centre_sampling_offset=10, patch_size=(64,128)):

    patches = [];

    # if no patches specifed just return original image

    if (sample_patches_to_generate == 0):
        return [img];

    # otherwise generate N sub patches

    else:

        # get all heights and widths

        img_height, img_width, _ = img.shape;
        patch_height = patch_size[1];
        patch_width = patch_size[0];

        # iterate to find up to N patches (0 -> N-1)

        for patch_count in range(sample_patches_to_generate):

            # if we are using centre weighted patches, first grab the centre patch
            # from the image as the first sample then take the rest around centre

            if (centre_weighted):

                # compute a patch location in centred on the centre of the image

                patch_start_h =  math.floor(img_height / 2) - math.floor(patch_height / 2);
                patch_start_w =  math.floor(img_width / 2) - math.floor(patch_width / 2);

                # for the first sample we'll just keep the centre one, for any
                # others take them from the centre position +/- centre_sampling_offset
                # in both height and width position

                if (patch_count > 0):
                    patch_start_h =  random.randint(patch_start_h - centre_sampling_offset, patch_start_h + centre_sampling_offset);
                    patch_start_w =  random.randint(patch_start_w - centre_sampling_offset, patch_start_w + centre_sampling_offset);

                # print("centred weighted path")

            # else get patches randonly from anywhere in the image

            else:

                # print("non centred weighted path")

                # randomly select a patch, ensuring we stay inside the image

                patch_start_h =  random.randint(0, (img_height - patch_height));
                patch_start_w =  random.randint(0, (img_width - patch_width));

            # add the patch to the list of patches

            patch = img[patch_start_h:patch_start_h + patch_height, patch_start_w:patch_start_w + patch_width]

            if (show_images_as_they_are_sampled):
                cv2.imshow("patch", patch);
                cv2.waitKey(5);

            patches.insert(patch_count, patch);

        return patches;

################################################################################

# add images from a specified path to the dataset, adding the appropriate class/type name
# and optionally adding up to N samples of a specified size with flags for taking them
# from the centre of the image only with +/- offset in pixels

def load_image_path(path, class_name, imgs_data, samples=0, centre_weighting=False, centre_sampling_offset=10 ,patch_size=(64,128)):

    # read all images at location

    imgs = read_all_images(path)

    img_count = len(imgs_data)
    for img in imgs:

        if (show_images_as_they_are_loaded):
            cv2.imshow("example", img);
            cv2.waitKey(5);

        # generate up to N sample patches for each sample image
        # if zero samples is specified then generate_patches just returns
        # the original image (unchanged, unsampled) as [img]

        for img_patch in generate_patches(img, samples, centre_weighting, centre_sampling_offset, patch_size):

            if show_additional_process_information:
                print("path: ", path, "class_name: ", class_name, "patch #: ", img_count)
                print("patch: ", patch_size, "from centre: ", centre_weighting, "with offset: ", centre_sampling_offset)

            # add each image patch to the data set

            img_data = ImageData(img_patch)
            img_data.set_class(class_name)
            imgs_data.insert(img_count, img_data)
            img_count += 1

    return imgs_data

################################################################################

# load image data from specified paths

def load_images(paths, class_names, sample_set_sizes, use_centre_weighting_flags, centre_sampling_offset=10, patch_size=(64,128)):
    imgs_data = []  # type: list[ImageData]

    # for each specified path and corresponding class_name and required number
    # of samples - add them to the data set

    for path, class_name, sample_count, centre_weighting in zip(paths, class_names, sample_set_sizes, use_centre_weighting_flags):
        load_image_path(path, class_name, imgs_data, sample_count, centre_weighting, centre_sampling_offset, patch_size)

    return imgs_data

################################################################################

# return the global set of bow histograms for the data set of images

def get_bow_histograms(imgs_data):

    samples = stack_array([[img_data.bow_histogram] for img_data in imgs_data])
    return np.float32(samples)

################################################################################

# return the global set of hog descriptors for the data set of images

def get_hog_descriptors(imgs_data):

    samples = stack_array([[img_data.hog_descriptor] for img_data in imgs_data])
    return np.float32(samples)

################################################################################

# return global the set of numerical class labels for the data set of images

def get_class_labels(imgs_data):
    class_labels = [img_data.class_number for img_data in imgs_data]
    return np.int32(class_labels)

################################################################################
