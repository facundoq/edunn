import numpy as np
from ..model import ModelWithoutParameters


class Flatten(ModelWithoutParameters):

    def forward(self, x:np.ndarray):
        y = {}
        self.set_cache(x.shape)
        ### YOUR IMPLEMENTATION START  ###
        y = x.reshape(x.shape[0], -1)
        ### YOUR IMPLEMENTATION END  ###
        return y

    def backward(self, δEδy:np.ndarray):
        δEδx = {}
        original_shape, = self.get_cache()
        ### YOUR IMPLEMENTATION START  ###
        δEδx = δEδy.copy().reshape(original_shape)
        ### YOUR IMPLEMENTATION END  ###
        return δEδx, {}