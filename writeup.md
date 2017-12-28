
## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/image1.png

---

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for HOG feature extraction can be found in the function `get_hog_features()` in the file `p5_helper_functions.py`).  When training, this function is invoked in `extract_features()` in the file `train.py`.

I used all the provided PNG image files for training: 8792 vehicle images and 8968 non-vehicle images. To reduce any bias during training I balanced vehicle vs. non-vehicle images by truncating the `notcar` array to match the length of the `car` array (line 69 in `train.py`). I randomly shuffled both sets of images before the truncation for best results.

#### 2. Explain how you settled on your final choice of HOG parameters.

I briefly explored different color spaces and different `skimage.hog()` parameters. Using the validation results from the training process, I found using 'ALL' HOG channels gave better results than any single channel. I also found `YCrCb` and `LUV` color spaces gave the best result. At this point, my validation results gave 100% correctness so I felt no need for additional tuning.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using spatial binning, color histogram, and HOG features (`bin_spatial()`, `color_hist()`, and `get_hog_features()` in `p5_helper_fcns()`).

Using a spatial bin size of 32x32, 32 bin histogram, and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` yielded a feature vector of size 8460. These color, spatial, and HOG features are concatenated in the function `extract_features()`. The feautre vector is normalized using `StandardScalar` (line 98 in `train.py`).

The linear SVM with all of the training parameters are saved in a pickle file for use later in the classification process (lines 133-144 in `train.py`).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used a modified version of the sliding window code provided in lesson 35. The structure of this code is largely dictated by the training image size (64x64) and cellular nature of HOG features. The scales were left as a parameter and the window offset fixed at 2 cells (75% overlap). 

Since window offset has a granualarity of a cell, the possible overlap values are n/8. I felt 4/8 and greater would lose too much detail and 1/8 would require excessive processing. 3/8 is a possibility but I stayed with 2/8.

The scales must also respect the integer-valued nature of the data. I selected scales of 1.0, 1.5, and 2.0 by estimating the size of cars in test images. These scales are adequate for near and middle distance cars, but too large for cars in the distance. I also adjusted the y search range based on scale - smaller cars higher in the frame.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Example output for all the test images can be found in `VehicleDetectionAndTracking.ipynb`.

An earlier version of my code composited the video frame with the heat map as well as the bounding boxes. Removing this feature improved performance by 50 per cent.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

An example from a test image is presented in the first cell of `VehicleDetectionAndTracking.ipynb`. The three images are displayed below. The first image shows the bounding boxes found for all three scales. The second image shows the thresholded heat map. In this case, the car in the opposing lane is culled. Finally, the bounding boxes from `scipy.ndimage.measurements.label()` are presented.

![alt test][image1]

My classifier did not seem to produce any false positives. However, it did produce false negatives. I used an averaging scheme to suppress the random drop outs. I used a `collections.deque` data structure to hold the bounding boxes from the previous four frames (line 29 in `P5.py`). I generated an aggregate heat map from this collection. This approach is reasonable for cars travelling in the same direction since their relative speed difference is small. The averaging does cause the bounding box to somewhat lag the target vehicle. I would not expect it to work well for oncoming traffic or a stalled vehicle.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 
- night? (more images)
- right edge (adjust search pattern)
- oncoming traffic (modify averaging scheme to account for relative velocity)
- stalled cars (same0)
- trucks or other non-car vehicles (more images)
-  

