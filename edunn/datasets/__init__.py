import numpy as np
import pathlib
from typing import Dict, Callable

from .. import utils

# basepath definition must be before importing regression and classification due to circular reference
basepath = pathlib.Path(__file__).parent.absolute()
from . import regression, classification


def shuffle_dataset(x, y):
    n = x.shape[0]
    indices = np.random.permutation(n)
    y = y[indices, :]
    x = x[indices, :]
    return x, y


def load_regression(dataset_name: str):
    return load(dataset_name, regression.loaders, "regression")


def load_classification(dataset_name: str, onehot=False):
    x, y, classes = load(dataset_name, classification.loaders, "classification")
    if onehot:
        y = utils.labels2onehot(y, len(classes))
    return x, y, classes


def load(dataset_name: str, loaders: Dict[str, Callable], title: str):
    if not dataset_name in loaders:
        raise ValueError(f"Unknown dataset {dataset_name}. Valid choices for {title} datasets:\n {loaders.keys()}.")

    dataset_loader = loaders[dataset_name]
    return dataset_loader()


def get_classification_names():
    return classification.loaders.keys()


def get_regression_names():
    return regression.loaders.keys()
