import warnings

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.decomposition import PCA


from src.utils.PrepareData import getTrainAndTestData

warnings.filterwarnings('ignore')


def kfold(X, y):
    skf = StratifiedKFold(n_splits=10)
    f1_test = []
    f1_train = []
    for train, test in skf.split(X, y):
        print(train)
        model = XGBClassifier()
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        f1_train.append(f1_score(model.predict(X[train]), y[train]))
        f1_test.append(f1_score(y_pred, y[test]))

    test_error = np.mean(f1_test)
    train_error = np.mean(f1_train)


    return train_error, test_error

def check_with_kfold(X,y):
    train_error, test_error = kfold(X, y)
    print("KFold = train_error = ", train_error)
    print("KFold = test_error = ", test_error)

    return 0



if __name__ == '__main__':

    test_strategy = 'kfold'

    print("getting the data")
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


    if('kfold' in test_strategy):
        check_with_kfold(np.concatenate([X_train,X_test], axis = 0), np.concatenate([y_train, y_test], axis = 0).ravel())
    else:
        model = XGBClassifier()
        print("building the model")
        model.fit(X_train, y_train)
        print("predicting now")
        y_pred = model.predict(X_test)
        print("f1 score on train dataset = ", f1_score(model.predict(X_train), y_train))
        print("f1 score on test dataset = ",f1_score(y_test, y_pred))



