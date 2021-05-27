import numpy as np
from ..model import ModelWithParameters

from ..initializers import Initializer

from .linear import Linear
from .bias import Bias


class LinearRegression(ModelWithParameters):
    '''
    A LinearRegression model applies a linear and bias function, in that order, to an input, ie
    y = wx+b, where w and b are the parameters of the Linear and Bias models,

    '''

    def __init__(self, input_size: int, output_size: int,
                 linear_initializer: Initializer = None, bias_initializer: Initializer = None, name=None):
        self.output_size=output_size
        self.input_size=input_size
        self.linear = Linear(input_size, output_size, initializer=linear_initializer)
        self.bias = Bias(output_size, initializer=bias_initializer)
        super().__init__(name=name)

    def forward(self, x: np.ndarray):
        # calculate and return bias(linear(x))
        y = np.zeros((x.shape[0],self.output_size)) # default value
        ### YOUR IMPLEMENTATION START  ###
        y_linear = self.linear.forward(x)
        y = self.bias.forward(y_linear)
        ### YOUR IMPLEMENTATION END  ###
        return y

    def backward(self, δEδy: np.ndarray):
        # Compute gradients for the parameters of the bias and linear models
        δEδbias, δEδlinear = {}, {}
        δEδx = np.zeros((δEδy.shape[0],self.input_size)) # default value

        ### YOUR IMPLEMENTATION START  ###
        δEδx_bias, δEδbias = self.bias.backward(δEδy)
        δEδx, δEδlinear = self.linear.backward(δEδx_bias)
        ### YOUR IMPLEMENTATION END  ###

        # combine gradients for parameters from linear and bias models
        # to obtain parameters for the Linear Regression (lr) model
        δEδlr = {**δEδbias, **δEδlinear}
        return δEδx, δEδlr

    def get_parameters(self):
        # returns the combination of parameters of linear and bias models
        # assumes these models don't use the same parameter names
        # ie: Linear has `w`, bias has `b`
        p_linear = self.linear.get_parameters()
        p_bias = self.bias.get_parameters()
        p = {**p_linear, **p_bias}
        return p
