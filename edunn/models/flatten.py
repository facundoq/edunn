import numpy as np
from ..model import Model


class Flatten(Model):

    def forward(self, x: np.ndarray):
        y = {}
        self.set_cache(x.shape)
        """ YOUR IMPLEMENTATION START """
        y = x.reshape(x.shape[0], -1)
        """ YOUR IMPLEMENTATION END """
        return y

    def backward(self, dE_dy: np.ndarray):
        dE_dx = {}
        original_shape, = self.get_cache()
        """ YOUR IMPLEMENTATION START """
        dE_dx = dE_dy.copy().reshape(original_shape)
        """ YOUR IMPLEMENTATION END """
        return dE_dx, {}
