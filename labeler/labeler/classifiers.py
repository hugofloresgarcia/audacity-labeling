import numpy as np
import torch
from joblib import load
import pandas as pd

def get_model(model_name):
    if 'svm' in model_name:
        model = LinearSVM(
            path_to_model='./classifiers/'+model_name
        )
    else:
        raise ValueError("incorrect classifier name")

    return model

class LinearSVM:
    def __init__(self, path_to_model):
        self.model = load(path_to_model + '_classifier')
        self.pca = load(path_to_model + '_pca')
        self.classes = list(pd.read_csv(path_to_model + '_classes.csv').columns)
        print(self.classes)

    def predict(self, x, ts=None):
        assert isinstance(x, np.ndarray), "input must be numpy array"
        assert x.ndim == 2, "input must be openl3 embedding with ndim 2"

        x = self.pca.transform(x)

        pred = self.model.predict(x)
        labels = [self.classes[int(i)] for i in pred]

        ts = ts

        return labels, ts