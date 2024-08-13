import numpy as np
from ..model import ModelWithoutParameters, Phase
from ..initializers import Initializer, RandomNormal


class Dropout(ModelWithoutParameters):

    def __init__(self, p: float = 0.5, name=None):
        super().__init__(name=name)
        self.p = p

    def forward(self, x: np.ndarray):
        u = np.zeros_like(x)
        """ YOUR IMPLEMENTATION START """
        # default: y = x
        u = np.random.binomial(1, self.p, size=x.shape) / self.p
        if self.phase == Phase.Training:
            y = x * u
        else:
            y = x
        """ YOUR IMPLEMENTATION END """
        self.set_cache(u)
        return y

    def backward(self, dE_dy: np.ndarray):
        dE_dx = {}
        # Retrieve u from cache
        u, = self.get_cache()
        """ YOUR IMPLEMENTATION START """
        dE_dx = dE_dy * u
        """ YOUR IMPLEMENTATION END """
        return dE_dx, {}
