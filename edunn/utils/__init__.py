from .comparisons import *
from . import check_gradient
from .numerical_gradient import numerical_gradient

import numpy as np

def onehot2labels(onehot):
    return onehot.argmax(axis=1)

def labels2onehot(labels,n_classes):
    n = len(labels)
    onehot = np.zeros((n, n_classes))
    for i in range(n):
        onehot[i,labels[i]] = 1
    return onehot