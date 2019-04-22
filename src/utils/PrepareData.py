import glob
import numpy as np

from src.utils.ReadData import read_data
from sklearn.utils import shuffle

def getTrainAndTestData():

    images_healthy_path1 = '/Users/aayush/Downloads/YE358311_Fender_apron/YE358311_Healthy/'
    images_healthy_path2 = '/Users/aayush/Downloads/YE358311_Fender_apron/YE358311_Healthy_augmented/'

    images_defect_path1 = '/Users/aayush/Downloads/YE358311_Fender_apron/YE358311_defects/YE358311_Crack_and_Wrinkle_defect/'
    images_defect_path2 = '/Users/aayush/Downloads/YE358311_Fender_apron/YE358311_defects/YE358311_Crack_and_Wrinkle_defect/'

    images_healthy = read_data(images_healthy_path)
    images_defect = read_data(images_defect_path)

    label_healthy = np.ones((images_healthy.shape[0], 1))
    label_defect = np.zeros((images_defect.shape[0], 1))

    X = np.concatenate([images_healthy, images_defect], axis = 0)

    y = np.concatenate([label_healthy, label_defect], axis = 0)

    X, y = shuffle(X, y)

    return X, y






