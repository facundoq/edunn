import numpy as np
from ..model import ModelWithoutParameters, Phase
from ..initializers import Initializer, RandomNormal


class Dropout(ModelWithoutParameters):

    def __init__(self, p: float = 0.5, name=None):
        super().__init__(name=name)
        self.p = p

    def forward(self, x: np.ndarray):
        u = np.zeros_like(x)
        """YOUR IMPLEMENTATION START"""
        if self.phase == Phase.Training:
            u = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            y = x * u
        else:
            u = np.ones_like(x)
            y = x
        """YOUR IMPLEMENTATION END"""
        self.set_cache(u)
        return y

    def backward(self, δEδy: np.ndarray):
        δEδx = {}
        # Retrieve u from cache
        (u,) = self.get_cache()
        """YOUR IMPLEMENTATION START"""
        δEδx = δEδy * u
        """YOUR IMPLEMENTATION END"""
        return δEδx, {}
