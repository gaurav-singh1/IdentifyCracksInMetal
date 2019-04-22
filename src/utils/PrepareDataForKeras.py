import warnings
import numpy as np
import h5py

from sklearn.utils import shuffle
from src.utils.PrepareData import getTrainAndTestData
from src.utils.ReadData import read_data
warnings.filterwarnings('ignore')


## preparing training data
X_train_healthy = read_data('/Users/aayush/Downloads/YE358311_Fender_apron/train/healthy/')
y_train_healthy = np.ones((X_train_healthy.shape[0], 1))

X_train_defective = read_data('/Users/aayush/Downloads/YE358311_Fender_apron/train/defective/')
y_train_defective = np.zeros((X_train_defective.shape[0], 1))

X_train = np.concatenate([X_train_healthy, X_train_defective], axis = 0)
y_train = np.concatenate([y_train_healthy, y_train_defective], axis = 0)

X_train, y_train = shuffle(X_train, y_train)



print("training data shape = ")
print("X_train = ",X_train.shape)
print("y_train = ",y_train.shape)

## preparing validation data

X_valid_healthy = read_data('/Users/aayush/Downloads/YE358311_Fender_apron/validation/healthy/')
y_valid_healthy = np.ones((X_train_healthy.shape[0], 1))

X_valid_defective = read_data('/Users/aayush/Downloads/YE358311_Fender_apron/validation/defective/')
y_valid_defective = np.zeros((X_train_defective.shape[0], 1))

X_valid = np.concatenate([X_valid_healthy, X_valid_defective], axis = 0)
y_valid = np.concatenate([y_valid_healthy, y_valid_defective], axis = 0)

X_valid, y_valid = shuffle(X_train, y_train)


print("validation data shape = ")
print("X_valid = ",X_valid.shape)
print("y_valid = ",y_valid.shape)



with h5py.File('/Users/aayush/PycharmProjects/Cracks/CracksImages.hdf5', 'w') as f:
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('X_test', data=X_valid)
    f.create_dataset('y_train', data=y_train)
    f.create_dataset('y_test', data=y_valid)



print("------------ Reading from h5.py file yields -----------------")


with h5py.File('/Users/aayush/PycharmProjects/Cracks/CracksImages.hdf5', 'r') as f:
    X_train = f['X_train']
    X_test = f['X_test']
    y_train = f['y_train']
    y_test = f['y_test']

    print("reading from h5 file")
    print(X_train.shape)
    print(y_train.shape)

