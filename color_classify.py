import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split

# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        # mpimg.imread gives RGB while cv2.imread give BGR
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

        else: feature_image = np.copy(image)
        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features)))
    # Return list of feature vectors
    return features


# Divide up into cars and notcars
# Read vehicle folders' names
folders_vehicle = glob.glob('vehicles/vehicles/*')
images_vehicle = []
for folder in folders_vehicle:
    image_names = folder + '/*.png'
    images_vehicle = images_vehicle +  glob.glob(image_names)

# Read non-vehicle folders' names
folders_non_vehicle = glob.glob('non-vehicles/non-vehicles/*')
images_non_vehicle = []
for folder in folders_non_vehicle:
    image_names = folder + '/*.png'
    images_non_vehicle = images_non_vehicle +  glob.glob(image_names)


# images_vehicle = glob.glob('vehicles/vehicles/GTI_Far/?????0001.png')
# images_non_vehicle = glob.glob('non-vehicles/non-vehicles/Extras/?????1.png')

cars = []
notcars = []

for image in images_vehicle:
    cars.append(image)

for image in images_non_vehicle:
    notcars.append(image)

print('Read {:} images of car'.format(len(cars)))
print('Read {:} images of notcar'.format(len(notcars)))

# TODO play with these values to see how your classifier
# performs under different binning scenarios
# spatial = 16
# histbin = 16

spatial = 12
histbin = 16

for histbin in range(4, 64, 4):
    print('------------- histbin = {:} -------------'.format(histbin))
    car_features = extract_features(cars, cspace='LUV', spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256))
    notcar_features = extract_features(notcars, cspace='LUV', spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256))

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

    print('Using spatial binning of:',spatial,
        'and', histbin,'histogram bins')
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
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

# Tests show that LUV color space has the best result with fixed spatial and histbin
# Tests show that spatial = 12 (16) has the best result with LUV and fixed histbin
# Tests show that histbin = 48 has the best result with LUV and spatial = 12
# LUV, spatial = 12 and histbin = 48 achieves 96% test accuracy
