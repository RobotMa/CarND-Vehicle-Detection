## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image0]: ./output_images/vehicle.png
[image1]: ./output_images/non_vehicle.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./output_images/sliding_windows.jpg "Sliding Windows"
[image41]: ./output_images/bbox_test1.jpg
[image42]: ./output_images/bbox_test3.jpg
[image43]: ./output_images/bbox_test4.jpg
[image44]: ./output_images/bbox_test5.jpg
[image5]: ./output_images/bounding_boxes_and_heatmap.png
[image6]: ./output_images/label.png
[image7]: ./output_images/final_bbox.png
[image8]: ./test_data/TestAccuracySpatialHistbin.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained from line 85 to 169 in the file called `hog_classify.py`. Note that all of the core functions are included through the file called `lesson_function.py` and this is also true for the later tests and pipeline.  

In the `hog_classify.py` file, I started by reading in all the `vehicle` and `non-vehicle` images.  Data augmentation is employed by flipping both the vehicle and non-vehicle images from the given data set as in the file `augment_data.py`. Note that this file needs to run for only once. The final non-vehicle data set has the folders `Extras`, `GTI`, `Extras_flipped` and `GTI_flipped`. The final vehicle data set constains the folders `GTI_far`, `GTI_Left`, `GTI_MiddleClose`, `GTI_Right`, `KITTI` and their flipped counterparts. The final data set gives a total of 17939 non-vehicle images and 17590 vehicle images. The augmented data set resulted in a better trained classifier on the test images in the folder `test_images`. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

Vehicle                    |   non-vehicle
:-------------------------:|:-------------------------:
![alt_text][image0]        |  ![alt_text][image1]

<!-- I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HSV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
-->

<!-- ![alt text][image2] -->

#### 2. Explain how you settled on your final choice of HOG parameters.

Before delving into the HOG parameters, extensive experiements were performed to pick out the best color features. The best color space, number of histbin and number of spatial were picked out by running experiment in `color_classify.py` in the folder `other_functions`. It can be observed that LUV color space performed the best when spatial and histbin were fixed. In addition, when LUV was used with histbin fixed, spatial = 12 (16) gave the highest test accuracy in general. Moreover, histbin = 48 performed almost the best when spatial = 12 and LUV color spaced was used. In conclusion, an ideal color feature is LUV, spatial = 12 and histbin = 48, and this combination resulted in 96% plus test accuracy. A complete diagram can be seen as follow. The next step comes down to the selection of HOG parameters.  

![alt text][image8]

Since LUV color space contains negative values and can't be applied to the built-in hog functions, several other color spaces were tested and HSV showed the highest test accuracy. Spatial and histbin features were included as well but under the `LUV` color space for their best performance.  Different channels of HSV were tested with fixed `pix_per_cell`, `cell_per_block` and `orient`. All of the features were normalized before training the classifier. Each of the hog parameters was tested with the other fixed as well, and the best parameters selected were `orient = 9`, `pix_per_cell=8`, `cell_per_block=2`, and `hog_channel = "ALL"`, which achieved 99.13% test accuracy. It is observed that `pix_per_cell` andi `cell_per_block` have relatively low impact on the final test accuracy.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
    
I trained a linear SVM using the augmented data set with features of spatial and histbin under `LUV` colorspace and hog under `HSV` colorspace. This part was done from Line 171 to 216 in `hog_classify.py` and the trained linear SVC along its parameters were saved as pickle files. The classifier can achieve 99% plus test accuracy repeatedly which reflected its good performance. However, false positives can still be detected in the video processing, and this can be mitigated by
introducing a bigger data set or deep learning based classifier. However, using heatmap for filtering the false positive given time series images performed well enough which will be shown later.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Many experiments were performed to tune the parameters for sliding window search. As shown from line 72 to 86 in the file named `pipeline.py`, four different sizes of sliding windows were defined along with their overlap rates and the area to search. The basic principle is to design windows that has significantly different sizes to cover potential smaller vehicles near the horizon and larger vehicles close to the bottom of the image. A visualization of the sliding windows can be seen
as follows:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?
 
Ultimately I searched on four sizes of window using HSV all-channel HOG features plus spatially binned color and histograms of color (LUV colorspace as mentioned before) in the feature vector, which provided a nice result. Note that not any false positive filtering technique was employed for the following four examples, and the false positive in the 4th image can be easily filtered using the technique that will be introduced in the next section. Here are the example images:

Example 1                  |   Example 2
:-------------------------:|:-------------------------:
![alt_text][image41]       |  ![alt_text][image42]

Example 3                  |   Example 4
:-------------------------:|:-------------------------:
![alt_text][image43]       |  ![alt_text][image44]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://www.youtube.com/watch?v=ZK4GcmxS-uY)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. The corrected threshold was obtained through a trial and error approach.   I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Rather than analyzing the normal cases, an edge case is picked out where two vehicles are very close to each other. It can be seen later that vehicle region can be detected but the pipeline can't differentiate the correct number of vehicles because two close vehicles are merged into one.
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

##### Here are six frames and their corresponding heatmaps:

![alt text][image5]

##### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

##### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

Note that in the final pipeline implementation, a deque of maximum length of 15 is created for the heatmap, such that almost all of the false positives in each frame in the project video can be filtered out in the final generated video. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

* As mentioned in the Video Implementation section, the pipeline fails to differentiate two vehicles when they are very close to each other. One possible way to solve this is to assign the detected region of different vehicles with different labels, which happens at the prediction stage.This will add a third dimension to the heatmap which not only highlights the region with the highest confidence of detecting vehicles, but also memorizes the type of vehicle that is being detected. This pre-labled heatmap can be used in combination of the current labeling function to further separate two vehicles that are close or overlaped.  

* Computation speed has become an issue and the current total time to process the project video amounts to over half an hour. This is can be solved by rewriting the code in c++ and introducing parallel programming. Also, extracting the HOG features just once for the entire image should also be promising, but due to the debugging difficulty deriving the pipeline, it is not adopted in this project.
