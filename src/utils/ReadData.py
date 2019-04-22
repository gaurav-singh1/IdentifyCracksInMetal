import glob

import cv2
import numpy as np


def get_image_paths(dir):
    image_paths = glob.glob(dir+'*.jpg')
    return image_paths

def resizeimage(img):
#     print(img.shape[1])
#     print(img.shape[0])
    scale_percent = 5  # percent of original size
    # print(scale_percent)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    resized/=255
    # print("re = ",resized.shape)
    return resized



def read_images_from_folder(dir):
    image_paths = get_image_paths(dir)
    images = np.array([np.array(resizeimage(cv2.imread(image_path))) for image_path in image_paths])
    return images



def read_images_as_features(dir):
    image_paths = get_image_paths(dir)
    images = np.array([np.array(resizeimage(cv2.imread(image_path, 0))).flatten() for image_path in image_paths])
    return images


def read_data(dir):
    images = read_images_from_folder(dir)
    return images




