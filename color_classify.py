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
from lesson_functions import *
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# from sklearn.cross_validation import train_test_split

# Divide up into cars and notcars
# Read vehicle folders' names
# Timing the image reading process
t1_read = time.time()
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
t2_read = time.time()
print('Spent {:} seconds to read in car and non-car images'.format(round(t2_read - t1_read)))

print('Read {:} images of car'.format(len(cars)))
print('Read {:} images of notcar'.format(len(notcars)))

# TODO play with these values to see how your classifier
# performs under different binning scenarios
# spatial = 16
# histbin = 16

spatial = range(4, 32, 4)
histbin = range(4, 64, 4)
len_s = len(spatial)
len_h = len(histbin)
data_matrix = np.zeros((len_s, len_h))

for i in range(len_s):
    for j in range(len_h):
        print('--- histbin = {:} and spatial = {:} -------------'.format(histbin[j], spatial[i]))
        car_features = extract_features(cars, color_space='LUV', spatial_size=(spatial[i], spatial[i]),
                hist_bins=histbin[j], hog_feat=False)
        notcar_features = extract_features(notcars, color_space='LUV', spatial_size=(spatial[i], spatial[i]),
                hist_bins=histbin[j], hog_feat=False)

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

        print('Using spatial binning of:',spatial[i],
                'and histogram bins of:', histbin[j])
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
        data_matrix[i][j] = round(svc.score(X_test, y_test), 4)
        # Check the prediction time for a single sample
        t=time.time()
        n_predict = 10
        print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
        print('For these',n_predict, 'labels: ', y_test[0:n_predict])
        t2 = time.time()
        print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

print(data_matrix)
# Tests show that LUV color space has the best result with fixed spatial and histbin
# Tests show that spatial = 12 (16) has the best result with LUV and fixed histbin
# Tests show that histbin = 48 has the best result with LUV and spatial = 12
# LUV, spatial = 12 and histbin = 48 achieves 96% test accuracy
