import numpy as np

from ..model import ErrorModel
import simplenn as sn

class BinaryCrossEntropyWithLabels(ErrorModel):

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

        E = np.zeros((n,1))
        ### COMPLETAR INICIO ###
        for i in range(n):
            miss = y_true[i] * y[i] + (1 - y_true[i]) * (1 - y[i])
            if miss==0:
                miss += sn.eps
            E[i] = - np.log(miss)
        # print(error)
        ### COMPLETAR FIN ###
        assert np.all(np.squeeze(E).shape == y_true.shape)
        self.set_cache(y_true,y)
        return E

    def backward(self, δEδyi):
        y_true,y = self.get_cache()
        δEδy = np.zeros_like(y)
        n,classes = y.shape
        ### COMPLETAR INICIO ###
        for i in range(n):
            miss = y_true[i] * y[i] - (1 - y_true[i]) * (1 - y[i])
            if miss==0:
                miss+=sn.eps
            δEδy[i] = - 1/miss
        ### COMPLETAR FIN ###

        return δEδy*δEδyi,{}

