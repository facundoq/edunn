import numpy as np
import pathlib
import random
basepath = pathlib.Path(__file__).parent.absolute()

def shuffle_dataset(x,y):
    n = x.shape[0]
    indices = np.random.permutation(n)
    y = y[indices,:]
    x = x[indices,:]
    return x,y

def load_regression_dataset(filename,shuffle=True):
    data=np.loadtxt(basepath / filename,skiprows=1,delimiter=",")
    x = data[:,:-1]
    y = data[:,-1:]
    if shuffle:
        x, y = shuffle_dataset(x,y)
    return x,y

def load_classification_dataset(filename,classes,shuffle=True):
    data=np.loadtxt(basepath / filename,skiprows=1,delimiter=",")
    x = data[:,:-1]
    y = data[:,-1:]
    if shuffle:
        x, y = shuffle_dataset(x,y)
    y=y.astype(np.int)
    return x,y, classes

def load_study_regression(): return load_regression_dataset("study_regression.csv")

def load_study_classification2d_easy(): return load_classification_dataset("study_logistic_2d_easy.csv",["Desaprobado","Aprobado"])

def load_study_classification():
    data=np.loadtxt(basepath /"study_logistic.csv",skiprows=1,delimiter=",")
    x = data[:,0:1]
    y = data[:,-1:]
    y=y.astype(np.int)
    classes = ["Desaprobado","Aprobado"]
    return x,y,classes

def load_study_classification2d():
    data=np.loadtxt(basepath /"study_logistic_2d.csv",skiprows=1,delimiter=",")
    x = data[:,0:2]
    y = data[:,-1:]
    y=y.astype(np.int)
    classes = ["Desaprobado","Aprobado"]
    return x,y,classes

def load_iris():
    data=np.loadtxt(basepath /"iris.csv",skiprows=1,delimiter=",")
    x = data[:,0:4]
    y = data[:,-1:]
    y=y.astype(np.int)
    classes = ["setosa","versicolor","virginica"]
    return x,y,classes

def load_boston(): return load_regression_dataset("boston.csv")

def red_wine(): return load_regression_dataset("winequality_red.csv")

def white_wine(): return load_regression_dataset("winequality_white.csv")


names = {
    "iris":load_iris,
    "study_1d":load_study_classification,
    "study_2d":load_study_classification2d,
    "study_2d_easy":load_study_classification2d_easy,
    "study_regression_1d":load_study_regression,
    "boston":load_boston,
    "red_wine":load_boston,
    "white_wine":load_boston,
}

def load(dataset_name:str):
    if not dataset_name in names:
        raise ValueError(f"Unknown dataset {dataset_name}. Check datasets.names for valid choices.")

    dataset_loader = names[dataset_name]
    return dataset_loader()
