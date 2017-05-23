import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from lesson_functions import *
from sklearn.externals import joblib
import glob
from collections import deque

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

images_names = glob.glob('test_images/project*.png')

# Parameters for efficient sliding window search
s1 = 64
s2 = 96
s3 = 128
s4 = 160
amp = 1.4
rate1 = 0.85
rate2 = 0.8
rate3 = 0.7
rate4 = 0.6
xy_window_list = [(s1, s1), (s2, s2), (s3, s3), (s4, s4)]
x_start_stop_list = [[None, None],[None, None],[None, None], [None, None]]
y_start_stop_list = [[400, 400 + int(amp*s1)],[400, 400 + int(amp*s2)],
                    [400, 400 + int(amp*s3)], [400, 400 + int(amp*s4)]]
xy_overlap_list = [(rate1, rate1), (rate2, rate2), (rate3, rate3), (rate4, rate4)]


fig, ax = plt.subplots(6, 2, figsize = (12,24))
ind = 0

heat_deque = deque(maxlen=6)

for image_name in images_names:

    image = mpimg.imread(image_name)

    draw_image = np.copy(image)

    windows = multiple_windows(image, xy_window_list, x_start_stop_list, y_start_stop_list,
                               xy_overlap_list)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    # plt.imshow(window_img)
    # plt.savefig('search-window-bbox-example-image-detected.jpg')

    # Read in image similar to one shown above
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    heat_deque.append(heat)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2.3)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    ind = ind + 1
    ind_2 = ind*2
    ind_1 = ind_2 - 1

    # Save the last image
    if ind == 6:
        final_image = np.copy(image)

    fig.add_subplot(6, 2, ind_1)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    fig.add_subplot(6, 2, ind_2)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    plt.axis('off')
    fig.tight_layout()

    # pure_image_name = image_name.rsplit('/',1)[1]
    # output_image_name = 'output_images/' + pure_image_name[:-4] + '_heated' + pure_image_name[-4:]

output_image_name = 'output_images/bounding_boxes_and_heatmap.png'
plt.savefig(output_image_name)

plt.gcf().clear()
heat_average = np.mean(heat_deque, 0)
heat_average_thresh = apply_threshold(heat_average, 2.3)
heatmap = np.clip(heat_average_thresh, 0, 255)
labels = label(heatmap)

label_image_name = 'output_images/label.png'
plt.imshow(labels[0], cmap='gray')
plt.savefig(label_image_name, bbox_inches='tight')

plt.gcf().clear()
final_image_name = 'output_images/final_bbox.png'
final_image = draw_labeled_bboxes(final_image, labels)
plt.imshow(final_image)
plt.savefig(final_image_name, bbox_inches='tight')

