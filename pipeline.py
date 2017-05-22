


def pipeline(image):

if __name__ == "__main__":

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

    draw_image = np.copy(image)

    xy_window_list = [(64, 64), (96, 96), (2*96, 2*96)]
    x_start_stop_list = [[None, None],[None, None],[None, None]]
    y_start_stop_list = [[400, 650],[400, 650],[400, 650]]
    xy_overlap_list = [(0.7, 0.7), (0.7, 0.7), (0.7, 0.7)]

    windows = multiple_windows(image, xy_window_list, x_start_stop_list, y_start_stop_list,
                               xy_overlap_list)

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    pickle.dump(hot_windows,  open("bbox_pickle.p", "wb"))

    plt.imshow(window_img)
    plt.savefig('search-window-bbox-example-image-detected.jpg')

