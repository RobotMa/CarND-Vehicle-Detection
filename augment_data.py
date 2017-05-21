# This file needs to be run for only once to produce flipped copies for car and non-car images
import glob
import os
import cv2

def produce_flipped_image(folders):

    current_path = os.getcwd()


    for folder in folders:
        images = []
        # Create a flipped folder in the same path of the original folder
        flipped_folder_path = current_path + '/' + folder + '_flipped'
        os.makedirs(flipped_folder_path)

        # Get the image names in the original folder
        image_names = folder + '/*.png'
        images = images +  glob.glob(image_names)

        for image_name in images:

            # Get the path of the image
            image_path = current_path + '/' + image_name
            image = cv2.imread(image_path)

            # Flip the image
            flipped_image = cv2.flip(image, 1)

            # Get pure image name (without any path information)
            pure_image_name = image_name.rsplit('/',1)[1]

            # Get the path of the flipped image
            flipped_image_path = flipped_folder_path + '/' +  pure_image_name[:-4] + '_flipped' + pure_image_name[-4:]
            cv2.imwrite(flipped_image_path, flipped_image)

# Divide up into cars and notcars
# Read vehicle folders' names
folders = glob.glob('vehicles/vehicles/*')
produce_flipped_image(folders)

# Read non-vehicle folders' names
folders = glob.glob('non-vehicles/non-vehicles/*')
produce_flipped_image(folders)
