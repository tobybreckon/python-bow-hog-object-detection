# Python Bag of (Visual) Words (BoW) and Histogram of Oriented Gradient (HOG) based Object Detection

An exemplar python implementation of the Bag of (Visual) Words and Histogram of Oriented Gradient (HOG) feature based object detection (BoW or HOG features, SVM classification) using [OpenCV](http://www.opencv.org).

Examples used for teaching within the undergraduate Computer Science programme
at [Durham University](http://www.durham.ac.uk) (UK) by [Prof. Toby Breckon](http://community.dur.ac.uk/toby.breckon/).

All tested with [OpenCV](http://www.opencv.org) 3.4.x and Python 3.x (requiring numpy also).

----

### OpenCV Setup:

To ensure you have your [OpenCV](http://www.opencv.org) setup correctly to use these in these examples - please follow the suite of testing examples [here](https://github.com/tobybreckon/python-examples-ip/blob/master/TESTING.md).

----

### Details:

You are provided with a set of 7 example files that can be run individually as follows:

- ```hog_training.py``` - performs object detection batch training using Histogram of Oriented Gradients (HOG) and SVM classification.

- ```hog_testing.py```  - performs object detection batch testing using Histogram of Oriented Gradients (HOG) and SVM classification.

- ```hog_detector.py``` - performs object detection via sliding window search using Histogram of Oriented Gradients (HOG) and SVM classification over a directory of specified images.

- ```bow_training.py``` - performs object detection batch training using a bag of visual words (BoW) approach and SVM classification.

- ```bow_testing.py``` - performs object detection batch testing using a bag of visual words (BoW) approach and SVM classification.

- ```bow_detector.py``` - performs object detection via sliding window search using a bag of visual words (BoW) approach and SVM classification over a directory of specified images.

- ```selective_search.py``` - performs selective search to generate object windows as an alternative to sliding window search (generates windows only, does not perform integrated object detection).

and additional supporting code in ```utils.py``` (image loading / feature extraction) and ```sliding_window.py``` (multi-scale sliding window) which are imported into the above.

----

### How to download and run:

Download each file as needed (or download/uncompress a zip from [here](https://github.com/tobybreckon/python-bow-hog-object-detection/archive/master.zip)) or to download the entire repository in an environment where you have git installed try:
```
git clone https://github.com/tobybreckon/python-bow-hog-object-detection
cd python-bow-hog-object-detection
```
In order to perform training you will need to first download the dataset, which can be achieved as follows on a linux/unix based system (or can alteratively be downloaded from [here](ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar) - ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar )
```
sh ./download-data.sh
```
If you run into errors such as _"libpng error: IDAT: invalid distance too far back"_ when running the commands below you may need to also run:
```
sh ./fix-pngs.sh
```
[Durham Students - just download the data sets from the [DUO](http://duo.dur.ac.uk) to avoid this.]

To perform training of the bag of works approach you can simply run as follows (or alternatively how you run python scripts in your environment):
```
python3 ./bow_training.py
```
whichs will perform the stages of loading image training set, feature descriptor extraction, k-means clustering and SVM classifier training and output two resulting files ```svm_bow.xml``` (the trained SVM classifier) and ```bow_dictionary.npy``` (the BoW set of visual codewords / cluster centres - known as the dictionary).

To perform batch testing of the bag of works approach you can then simply use (or alternatively ...):
```
python3 ./bow_testing.py
```
which will load the ```svm_bow.xml``` and ```bow_dictionary.npy``` created from training and report statistical testing performance over an independent set of test images (not used during training).

To perform detection over a set of images you can then simply use (or alternatively ...):
```
python3 ./bow_detector.py
```
which will again load the ```svm_bow.xml``` and ```bow_dictionary.npy``` created from training but now perform multi-scale sliding window based detection over a set of images in a directory specified by the variable ```directory_to_cycle = "...."``` at the top of this python script file.

The above instructions can be repeated for the set of ```hog_...py``` examples to perform training (to produce a single ```svm_hog.xml``` file), testing and subsequent detection as before.

----

### References

This code base was informed by the research work carried out in the following publications:

- [On using Feature Descriptors as Visual Words for Object Detection within X-ray Baggage Security Screening](http://community.dur.ac.uk/toby.breckon/publications/papers/kundegorski16xray.pdf) (M.E. Kundegorski, S. Akcay, M. Devereux, A. Mouton, T.P. Breckon), In Proc. International Conference on Imaging for Crime Detection and Prevention, IET, pp. 12 (6 .)-12 (6 .)(1), 2016. [[pdf](http://community.dur.ac.uk/toby.breckon/publications/papers/kundegorski16xray.pdf)] [[doi](http://dx.doi.org/10.1049/ic.2016.0080)]

- [Real-time Classification of Vehicle Types within Infra-red Imagery](http://community.dur.ac.uk/toby.breckon/publications/papers/kundegorski16vehicle.pdf) (M.E. Kundegorski, S. Akcay, G. Payen de La Garanderie, T.P. Breckon), In Proc. SPIE Optics and Photonics for Counterterrorism, Crime Fighting and Defence, SPIE, Volume 9995, pp. 1-16, 2016. [[pdf](http://community.dur.ac.uk/toby.breckon/publications/papers/kundegorski16vehicle.pdf)] [[doi](http://dx.doi.org/10.1117/12.2241106)]

- [A Photogrammetric Approach for Real-time 3D Localization and Tracking of Pedestrians in Monocular Infrared Imagery](http://community.dur.ac.uk/toby.breckon/publications/papers/kundegorski14photogrammetric.pdf) (M.E. Kundegorski, T.P. Breckon], In Proc. SPIE Optics and Photonics for Counterterrorism, Crime Fighting and Defence, SPIE, Volume 9253, No. 01, pp. 1-16, 2014. [[pdf](http://community.dur.ac.uk/toby.breckon/publications/papers/kundegorski14photogrammetric.pdf)] [[doi](http://dx.doi.org/10.1117/12.2065673)]

----

**Acknowledgements:** originally forked from an earlier Bag of Visual Words only version at https://github.com/siphomateke/PyBOW with the additional HOG and selective search code added to this newer version.

_[ but it appears some code portions may have broader origins elsewhere, such as from this tutorial - https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/ ]_

**Bugs:** _I do not claim this code to be bug free._ If you find any bugs raise an issue (or much better still submit a git pull request with a fix) - toby.breckon@durham.ac.uk

_"may the source be with you"_ - anon.
