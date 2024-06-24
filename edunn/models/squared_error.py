import numpy as np

from ..model import ModelWithoutParameters


class SquaredError(ModelWithoutParameters):
    def forward(self, y_true: np.ndarray, y: np.ndarray):
        delta = (y - y_true)

        n = y_true.shape[0]
        E = np.zeros((n, 1))
        ### YOUR IMPLEMENTATION START  ###
        E = np.sum(delta ** 2, axis=1, keepdims=True)
        ### YOUR IMPLEMENTATION END  ###
        self.set_cache(delta)
        return E

    def backward(self, dE_dEi):
        delta, = self.get_cache()
        # Calculate error w.r.t
        # y (the output of the model) and not y_true (which is a fixed value)
        dE_dy = np.zeros_like(delta)
        ### YOUR IMPLEMENTATION START  ###
        dEi_dy = 2 * delta
        dE_dy = dEi_dy * dE_dEi
        ### YOUR IMPLEMENTATION END  ###
        return dE_dy, {}
