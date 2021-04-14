import numpy as np
from . import basepath

classification_basepath=basepath/ "classification"

def load_classification_dataset(filename,classes):
    data=np.loadtxt(classification_basepath / filename,skiprows=1,delimiter=",")
    x = data[:,:-1]
    y = data[:,-1:]
    y=y.astype(int)
    y = np.squeeze(y)
    return x,y, classes


study2d_easy = lambda: load_classification_dataset("study2d_easy.csv", ["Failed", "Passed"])
study2d = lambda: load_classification_dataset("study2d.csv", ["Failed", "Passed"])
study1d = lambda: load_classification_dataset("study1d.csv", ["Failed", "Passed"])
iris = lambda: load_classification_dataset("iris.csv", ["setosa", "versicolor", "virginica"])

loaders = {
    "iris":iris,
    "study1d":study1d,
    "study2d":study2d,
    "study2d_easy":study2d_easy,
}