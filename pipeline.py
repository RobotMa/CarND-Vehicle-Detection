from moviepy.editor import VideoFileClip
from lesson_functions import *
import pickle
from sklearn.externals import joblib
from collections import deque

class HeatMap:
    def __init__(self):
        self.heat_map = deque(maxlen=15)

# Pipeline to detect the vehicles in a video clip
def pipeline(image):

    draw_image = np.copy(image)

    windows = multiple_windows(image, xy_window_list, x_start_stop_list, y_start_stop_list,
                           xy_overlap_list)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    Heat_Map.heat_map.append(heat)

    heat = np.mean(Heat_Map.heat_map, 0)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,2.3)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img

if __name__ == "__main__":

    # Load the trained clf as in hog_classify.py
    svc = joblib.load('trainedSVC.pkl')

    # Loae the parameters for the clf
    dist_pickle = pickle.load( open("svc_param.p", "rb" ) )
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

    Heat_Map = HeatMap()

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

    opt = 2
    if opt == 1:
        video = 'detected_test_video.mp4'
        clip = VideoFileClip("test_video.mp4")
        completed_clip = clip.fl_image(pipeline)
        completed_clip.write_videofile(video, audio=False)

    elif opt == 2:
        video = 'detected_project_video.mp4'
        clip = VideoFileClip("project_video.mp4")
        completed_clip = clip.fl_image(pipeline)
        completed_clip.write_videofile(video, audio=False)

    elif opt == 3:
        video = 'debug_project_video_43_46.mp4'
        clip = VideoFileClip("project_video.mp4").subclip(43, 46)
        completed_clip = clip.fl_image(pipeline)
        completed_clip.write_videofile(video, audio=False)

    elif opt == 4:
        video = 'debug_project_video_7_8.mp4'
        clip = VideoFileClip("project_video.mp4").subclip(7, 8)
        completed_clip = clip.fl_image(pipeline)
        completed_clip.write_videofile(video, audio=False)

