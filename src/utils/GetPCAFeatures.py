
from sklearn.preprocessing import StandardScaler

from src.utils.PrepareData import getTrainAndTestData


def scale_features(data):
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)



def pca_extractor(data):
    pca = PCA(0.99)
    pca.fit(data)
    print("pca.n_components = ",pca.n_components)
    print("pca.n_components_ = ",pca.n_components_)
    data_pca_transformed = pca.transform(data)

    return data_pca_transformed


def pca_features(data):
    data = scale_features(data)
    data = pca_extractor(data)


    return data
















