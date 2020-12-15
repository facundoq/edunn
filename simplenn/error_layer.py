import numpy as np

from simplenn.layer import ErrorLayer,SampleErrorLayer
import simplenn as sn


def fix_probabilities(p:np.ndarray):
    ## add a small value positive to avoid division by zero
    ## In terms of the model output, this equates to smoothing
    ## the probabilities, ie, adding a small non zero probability
    ## to all outputs
    p= p + sn.eps
    p/=p.sum(axis=1, keepdims=True)
    return p


class SquaredError(SampleErrorLayer):
    def forward(self, y_true:np.ndarray, y:np.ndarray):
        return np.sum((y - y_true) ** 2, axis=0)

    def backward(self, y_true:np.ndarray, y:np.ndarray):
        δEδy = 2 * (y - y_true)
        return δEδy,{}






class BinaryCrossEntropyWithLabels(SampleErrorLayer):

    '''
    Returns the CrossEntropy between two binary class distributions
    Receives a matrix of probabilities Nx1 for each sample, with the probability
     for class 1, ie if p=0.7, this indicates P(C=1) =0.7 and P(C=0)=0.3
    Receives a a vector of class labels, also one for each sample, with values either 0 or 1.

'''

    def forward(self, y_true:np.ndarray, y:np.ndarray):
        y_true = np.squeeze(y_true)
        assert(len(y_true.shape) == 1)
        assert y.min() >= 0

        n,c=y.shape

        error = np.zeros((n))
        ### COMPLETAR INICIO ###
        for i in range(n):
            miss = y_true[i] * y[i] + (1 - y_true[i]) * (1 - y[i])
            if miss==0:
                miss += sn.eps
            error[i] = - np.log(miss)
        # print(error)
        ### COMPLETAR FIN ###
        assert np.all(error.shape == y_true.shape)

        return error

    def backward(self, y_true:np.ndarray, y:np.ndarray):

        δEδy = np.zeros_like(y)
        n,classes = y.shape
        ### COMPLETAR INICIO ###
        for i in range(n):
            miss = y_true[i] * y[i] - (1 - y_true[i]) * (1 - y[i])
            if miss==0:
                miss+=sn.eps
            δEδy[i] = - 1/miss
        ### COMPLETAR FIN ###

        return δEδy,{}


class CrossEntropyWithLabels(SampleErrorLayer):
    '''
        Returns the CrossEntropy between two class distributions
        Receives a matrix of probabilities NxC for each sample and a vector
        of class labels, also one for each sample,.
        For each sample, this layer receives a vector of probabilities for each class,
        (which must sum to 1)
        and the label of the sample (which implies that class has an
        expected probability of 1, and 0 for the rest)
    '''

    ### Ayuda para implementar:
    ### http://facundoq.github.io/guides/crossentropy_derivative.html
    def forward(self, y_true:np.ndarray, y:np.ndarray):
        y_true = np.squeeze(y_true)

        assert(len(y_true.shape) == 1)
        assert y.min() >= 0

        n,c=y.shape

        #y_pred=fix_probabilities(y_pred)

        E = np.zeros((n))
        ### COMPLETAR INICIO ###
        for i in range(n):
            probability = y[i, y_true[i]]
            E[i] = -np.log(probability)
        ### COMPLETAR FIN ###
        assert np.all(E.shape == y_true.shape)

        return E

    def backward(self, y_true:np.ndarray, y:np.ndarray):

        #y_pred=fix_probabilities(y_pred)

        δEδy = np.zeros_like(y)
        n,classes = y.shape
        ### COMPLETAR INICIO ###
        for i in range(n):
            p=y[i, y_true[i]]
            δEδy[i, y_true[i]] = -1 / p
        ### COMPLETAR FIN ###
        return δEδy,{}
