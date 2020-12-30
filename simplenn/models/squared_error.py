import numpy as np

from ..model import ModelWithParameters

class SquaredError(ModelWithParameters):
    def forward_with_cache(self, y_true:np.ndarray, y:np.ndarray):
        delta = (y - y_true)
        ## COMPLETAR INICIO
        E = np.sum(delta** 2, axis=1)
        ## COMPLETAR FIN
        cache = (delta,)
        return E,cache

    def backward(self, δEδyi,cache):

        delta, = cache
        # Calculate error w.r.t
        # y (the output of the model) and not y_true (which is a fixed value)
        ## COMPLETAR INICIO
        δEδy = 2 * delta
        ## COMPLETAR FIN

        return δEδy* δEδyi,{}
