import numpy as np

from ..model import ErrorModel

class SquaredError(ErrorModel):
    def forward(self, y_true:np.ndarray, y:np.ndarray):
        delta = (y - y_true)

        n = y_true.shape[0]
        E = np.zeros((n,1))
        ### YOUR IMPLEMENTATION START  ###
        E = np.sum(delta** 2, axis=1,keepdims=True)
        ### YOUR IMPLEMENTATION END  ###
        self.set_cache(delta)
        return E

    def backward(self, δEδEi):
        delta, = self.get_cache()
        # Calculate error w.r.t
        # y (the output of the model) and not y_true (which is a fixed value)
        δEδy=np.zeros_like(delta)
        ### YOUR IMPLEMENTATION START  ###
        δEiδy = 2 * delta
        δEδy = δEiδy*δEδEi
        ### YOUR IMPLEMENTATION END  ###
        return δEδy,{}
