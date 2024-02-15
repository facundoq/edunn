import numpy as np
from . import utils

def accuracy(y_true:np.ndarray,y_pred:np.ndarray)->float:
    check_label_array(y_true)
    check_label_array(y_pred)
    return np.mean(y_pred==y_true)

def rmse(y_true:np.ndarray,y_pred:np.ndarray)->float:
    return np.sqrt(((y_true-y_pred)**2).sum(axis=1)).mean()

def mae(y_true:np.ndarray,y_pred:np.ndarray)->float:
    return np.abs(y_true-y_pred).mean(axis=0).mean()


def check_label_array(y:np.ndarray):
    if  len(y.shape)!=1:
        raise ValueError(f"True labels array must have a single dimension (shape is {y.shape} instead)")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f"True labels array must be of type int (type is {y.dtype} instead)")

def check_binary(y:np.ndarray):
    mi,ma=y.min(),y.max()
    if not (0<=mi and ma<=1):
        raise ValueError(f"Binary labels must have value 0 or 1, found values from {mi} to {ma} instead.")

def precision(y_true:np.ndarray,y_pred:np.ndarray)->float:
    check_label_array(y_true)
    check_label_array(y_pred)
    check_binary(y_true)
    check_binary(y_pred)
    n = len(y_true)
    pred_true_indices = y_pred == 1
    true_positives = np.logical_and(y_true == 1,y_pred ==1)

    return np.sum(true_positives)/np.sum(pred_true_indices)

def recall(y_true:np.ndarray,y_pred:np.ndarray)->float:
    check_label_array(y_true)
    check_label_array(y_pred)
    check_binary(y_true)
    check_binary(y_pred)


    true_indices = y_true == 1
    true_positives = np.logical_and(y_true == 1,y_pred ==1)

    return np.sum(true_positives)/np.sum(true_indices)

def fscore(y_true:np.ndarray,y_pred:np.ndarray)->float:
    check_label_array(y_true)
    check_label_array(y_pred)
    check_binary(y_true)
    check_binary(y_pred)
    p,r = precision(y_true,y_pred),recall(y_true,y_pred)
    return 2*p*r/(p+r)

def confusion(y_true:np.ndarray,y_pred:np.ndarray)->np.ndarray:
    check_label_array(y_true)
    check_label_array(y_pred)

    classes = y_true.max()+1
    c = np.zeros((classes,classes),dtype=int)
    n = len(y_true)
    for i in range(n):
        c[y_true[i],y_pred[i]]+=1
    return c

def classification_summary_onehot(y_true:np.ndarray,y_pred:np.ndarray):
    assert(len(y_true.shape)==2)
    assert(np.all(y_true.shape==y_pred.shape))
    y_true=utils.onehot2labels(y_true)
    y_pred=utils.onehot2labels(y_pred)
    classification_summary(y_true,y_pred)

def classification_summary(y_true:np.ndarray,y_pred:np.ndarray):
    classes = y_true.max()+1

    print(f"Accuracy: {accuracy(y_true,y_pred)} ({classes} classes)")
    if classes==2:
        p,r,f = precision(y_true,y_pred),recall(y_true,y_pred),fscore(y_true,y_pred)
        print(f"Precision: {p}")
        print(f"Recall: {r}")
        print(f"F-score: {f}")
    print(f"Confusion matrix: (rows true, columns pred)")
    c = confusion(y_true,y_pred)
    print(c)


def regression_summary(y_true:np.ndarray,y_pred:np.ndarray):
    print(f"RMSE {rmse(y_true,y_pred)}")
    print(f"MAE {mae(y_true,y_pred)}")