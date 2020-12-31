import numpy as np

from ..model import ErrorModel

class SquaredError(ErrorModel):
    def forward(self, y_true:np.ndarray, y:np.ndarray):
        delta = (y - y_true)

        n = y_true.shape[0]
        E = np.zeros((n,1))
        ## COMPLETAR INICIO
        E = np.sum(delta** 2, axis=1,keepdims=True)
        ## COMPLETAR FIN
        self.set_cache(delta)
        return E

    def backward(self, δEδEi):
        delta, = self.get_cache()
        # Calculate error w.r.t
        # y (the output of the model) and not y_true (which is a fixed value)
        δEδy=np.zeros_like(delta)
        ## COMPLETAR INICIO
        δEiδy = 2 * delta
        δEδy = δEiδy*δEδEi
        ## COMPLETAR FIN
        return δEδy,{}
