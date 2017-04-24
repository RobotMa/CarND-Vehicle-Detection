import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0, vis=False):
    # Create a list to append feature vectors to
    features = []
    if vis == True:
        hog_images = []

    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
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

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                if vis == False:
                    hog_feature = get_hog_features(feature_image[:,:,channel],
                                                    orient, pix_per_cell, cell_per_block,
                                                    vis, feature_vec=True)
                    hog_features.append(hog_feature)

                else:
                    hog_feature, hog_image = get_hog_features(feature_image[:,:,channel],
                                                            orient, pix_per_cell, cell_per_block,
                                                            vis, feature_vec=True)
                    hog_features.append(hog_feature)
                    hog_images.append(hog_image)

            hog_features = np.ravel(hog_features)
            features.append(hog_features)
        else:
            if vis == False:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                           pix_per_cell, cell_per_block, vis, feature_vec=True)
                features.append(hog_features)
            else:
                hog_features, hog_images = get_hog_features(feature_image[:,:,hog_channel], orient,
                           pix_per_cell, cell_per_block, vis, feature_vec=True)
                features.append(hog_features)


    if vis == False:
        return features
    else:
        return features, hog_images

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

# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
subsampling = False
if subsampling == True:
    sample_size = 500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
# bug  when using 'LUV': feature_image in the extract_feature
# function has negative values
# colorspace = 'LUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
colorspace = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
vis = True

import ipdb; ipdb.set_trace() #
t=time.time()

if vis == False:
    car_features = extract_features(cars, cspace=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel, vis=vis)
else:
    car_features, car_images = extract_features(cars, cspace=colorspace, orient=orient,
                                 pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, vis=vis)
if vis == False:
    notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel, vis=vis)
else:
    notcar_features, notcar_images = extract_features(notcars, cspace=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel, vis=vis)

t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
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

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
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
print('    My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
