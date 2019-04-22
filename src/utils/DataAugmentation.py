import random
import skimage as sk
from skimage import transform
from skimage import util
from skimage import exposure
import numpy as np
from scipy import ndimage
from skimage import io

import os


def color_inversion(image_array):
    color_inversion_image = util.invert(image_array)
    return color_inversion_image

def rescale_intensity(image_array):
    v_min, v_max = np.percentile(image_array, (0.2, 99.8))
    better_contrast = exposure.rescale_intensity(image_array, in_range=(v_min, v_max))
    return better_contrast

def gamma_correction(image_array):
    adjusted_gamma_image = exposure.adjust_gamma(image_array, gamma=0.4, gain=0.9)
    return adjusted_gamma_image

def log_correction(image_array):
    log_correction_image = exposure.adjust_log(image_array)
    return log_correction_image

def sigmoid_correction(image_array):
    sigmoid_correction_image = exposure.adjust_sigmoid(image_array)
    return sigmoid_correction_image


def random_rotation(image_array):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array):
    # add random noise to the image
    return sk.util.random_noise(image_array)

def horizontal_flip(image_array):
    return image_array[:, ::-1]

def vertical_flip(image_array):
    return image_array[::-1, :]

def blur_image(image_array):
    blured_image = ndimage.uniform_filter(image_array, size=(11, 11, 1))
    return blured_image


# dictionary of the transformations we defined earlier
available_transformations = {
    'rotate': random_rotation,
    'noise': random_noise,
    'horizontal_flip': horizontal_flip,
    'color_inversion': color_inversion,
    'rescale_intensity': rescale_intensity,
    'gamma_correction': gamma_correction,
    'log_correction': log_correction,
    'sigmoid_correction': sigmoid_correction,
    'random_noise': random_noise,
    'vertical_flip': vertical_flip,
    'blur_image': blur_image
}


#/Users/aayush/Downloads/YE358311_Fender_apron/validation/healthy
folder_path = '/Users/aayush/Downloads/YE358311_Fender_apron/validation/healthy'
folder_augmented_image = '/Users/aayush/Downloads/YE358311_Fender_apron/validation/healthy'

if(not os.path.exists(folder_augmented_image)):
    os.makedirs(folder_augmented_image)

num_files_desired = 10

# find all files paths from the folder
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and '.jpg' in f]
num_generated_files = 0

image_processed = 0

total_images = len(images)
print("total_images = ", total_images)
for image_path in images:
    # read image as an two dimensional array of pixels
    image_to_transform = sk.io.imread(image_path)
    image_name = image_path.split('/')[-1].split('.jpg')[0]
    # random num of transformation to apply
    for key in available_transformations:
        # num_transformations_to_apply = random.randint(1, len(available_transformations))
        # num_transformations = 0
        # transformed_image = None
        # random transformation to apply for a single image
        transformed_image = available_transformations[key](image_to_transform)
        new_file_path = '%s/augmented_image_%s.jpg' % (folder_augmented_image, image_name+key)
        # write image to the disk
        io.imsave(new_file_path, transformed_image)
        num_generated_files += 1
        print('total generated = ', num_generated_files)


    image_processed+=1
    print("{} of {}".format(image_processed, total_images))

