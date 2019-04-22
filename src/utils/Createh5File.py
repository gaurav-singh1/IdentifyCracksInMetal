import warnings

import h5py
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.PrepareData import getTrainAndTestData

warnings.filterwarnings('ignore')



X, y = getTrainAndTestData()
print("X, y obtained")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Now computing pca features on the dataset")
pca = PCA(0.99)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

with h5py.File('/Users/aayush/PycharmProjects/Cracks/CracksData.hdf5', 'w') as f:
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('X_test', data=X_test)
    f.create_dataset('y_train', data=y_train)
    f.create_dataset('y_test', data=y_test)


#
#
# with h5py.File('/Users/aayush/PycharmProjects/Cracks/CracksData.hdf5', 'r') as f:
#     X_train = f['X_train']
#     X_test = f['X_test']
#     y_train = f['y_train']
#     y_test = f['y_test']
#
#     print("reading from h5 file")
#     print(X_train.shape)
#     print(y_train.shape)
#
