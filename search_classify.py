import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from lesson_functions import *
from sklearn.externals import joblib


dist_pickle = pickle.load( open("svc_param.p", "rb" ) )
svc = joblib.load('trainedSVC.pkl')
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
color_space = dist_pickle["colorspace_hog"]

hog_channel = "ALL"
spatial_feat = True
hist_feat = True
hog_feat = True
y_start_stop = [400, 650] # Min and max in y to search in slide_window()

#image = mpimg.imread('bbox-example-image.jpg')
image = mpimg.imread('test_images/test5.jpg')
draw_image = np.copy(image)

# find_car approach
scale = 1.5
out_img = find_cars(image, y_start_stop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)

# plt.imshow(out_img)
# plt.savefig('find-car-bbox-example-image-detected.jpg')

# Parameters for efficient sliding window search
s1 = 64
s2 = 96
s3 = 128
s4 = 160
amp = 1.2
rate1 = 0.85
rate2 = 0.8
rate3 = 0.7
rate4 = 0.6
xy_window_list = [(s1, s1), (s2, s2), (s3, s3), (s4, s4)]
x_start_stop_list = [[None, None],[None, None],[None, None], [None, None]]
y_start_stop_list = [[400, 400 + int(amp*s1)],[400, 400 + int(amp*s2)],
                    [400, 400 + int(amp*s3)], [400, 400 + int(amp*s4)]]
xy_overlap_list = [(rate1, rate1), (rate2, rate2), (rate3, rate3), (rate4, rate4)]

windows = multiple_windows(image, xy_window_list, x_start_stop_list, y_start_stop_list,
                           xy_overlap_list)

# Plot the sliding windows
sliding_window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)
# plt.imshow(sliding_window_img)
# plt.savefig('output_images/sliding_windows.jpg', bbox_inches='tight')

hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

pickle.dump(hot_windows,  open("bbox_pickle.p", "wb"))

plt.imshow(window_img)
plt.savefig('output_images/bbox_test5.jpg', bbox_inches='tight')

