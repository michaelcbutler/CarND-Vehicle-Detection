import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from random import shuffle

from p5_helper_fcns import *


def extract_features(png_list, cspace='RGB', spatial_size=(32, 32), 
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0):
                     #spatial_feat=True, hist_feat=True, hog_feat=True):
    """Extract requested features from list of png files"""
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for png in png_list:
        #file_features = []
        image = mpimg.imread(png)
        # Color conversion if cspace other than 'RGB'
        feature_image = color_convert(image, cspace)

        # Compute spatial features if flag is set
        #if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
            # Append features to list
            #file_features.append(spatial_features)
        # Compute histogram features if flag is set
        #if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
            # Append features to list
            #file_features.append(hist_features)
        # Compute HOG features if flag is set
        #if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append features to list
            #file_features.append(hog_features)
        file_features = np.hstack((spatial_features, hist_features, hog_features))
        #features.append(np.concatenate(file_features))
        features.append(file_features)
    
    # Return list of feature vectors
    return features



# png training set
image_format = 'png'
cars = glob.glob('test_images/vehicles/*/*.png')
notcars = glob.glob('test_images/non-vehicles/*/*.png')

# jpeg training set
#image_format = 'jpeg'
#cars = glob.glob('test_images/vehicles_smallset/*/*.jpeg')
#notcars = glob.glob('test_images/non-vehicles_smallset/*/*.jpeg')
print("image format:", image_format)
print("number of car images:", len(cars))
print("number of notcar images:", len(notcars))

# randomize order
shuffle(cars)
shuffle(notcars)

# Balance cars and notcars
sample_size = min(len(cars), len(notcars)) # balance cars vs notcars
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

print("Sample size =", sample_size)

### TODO: Tweak these parameters and see how the results change.
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (32, 32)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

t=time.time()
car_features = extract_features(cars, cspace=colorspace, 
                                spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel)
notcar_features = extract_features(notcars, cspace=colorspace, 
                                   spatial_size=spatial_size, hist_bins=hist_bins, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                   hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)         
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print("orient = {}, pix_per_cell = {}, cells_per_block = {}, spatial_size = {}, hist_bins = {}, colorspace = {}".format(
    orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, colorspace))
print("hog channel = {}, image_format = {}".format(hog_channel, image_format))

print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
print('My SVC predicts:     ', svc.predict(X_test[0:n_predict]))
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# Save SVC, X_scalar, and parameters to file
dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
dist_pickle["colorspace"] = colorspace
dist_pickle["hog_channel"] = hog_channel
dist_pickle["image_format"] = image_format
pickle.dump( dist_pickle, open( "p5_pickle.p", "wb" ))