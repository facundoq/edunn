import edunn as nn
import numpy as np


class FakeModel(nn.ModelWithParameters):
    # Fake model with a single parameter array with a parameter items
    # Parameters always initialized as `parameter`
    # Derivative of parameters is always `gradient`
    # Derivative of input is always 0
    def __init__(self, parameter: np.ndarray, gradient: np.ndarray, name=None):
        super().__init__(name=name)
        self.parameter = parameter
        self.gradient = gradient
        assert (np.all(parameter.shape == gradient.shape))
        self.register_parameter("parameter", self.parameter)

    def forward(self, x: np.ndarray):
        return x

    def backward(self, dE_dy: np.ndarray):
        dE_dx = 0
        dE_dp = {"parameter": self.gradient}
        return dE_dx, dE_dp


class FakeError(nn.ModelWithoutParameters):
    def __init__(self, error=1, derivative_value=1, name=None):
        super().__init__(name=name)
        self.error = error
        self.derivative_value = derivative_value

    # Fake error function without parameters
    # forward always returns self.error
    # backward always returns array with same shape as input
    # and all values set to self.derivative_value
    def forward(self, y, y_true):
        self.set_cache(y.shape)
        return self.error

    def backward(self, dE: float):
        shape, = self.get_cache()
        return np.ones(shape) * self.derivative_value, {}
