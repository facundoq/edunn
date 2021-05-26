import simplenn as sn
import numpy as np


class FakeModel(sn.ModelWithParameters):
    # Fake model with a single parameter array with a parameter items
    # Parameters always initialized as `parameter`
    # Derivative of parameters is always `gradient`
    # Derivative of input is always 0
    def __init__(self,parameter:np.ndarray,gradient:np.ndarray):
        super().__init__()
        self.parameter = parameter
        self.gradient = gradient
        assert (np.all(parameter.shape==gradient.shape))
        self.register_parameter("parameter", self.parameter)

    def forward(self, x:np.ndarray):
        return x

    def backward(self, δEδy:np.ndarray):
        δEδx = 0
        δEδp = {"parameter": self.gradient}
        return δEδx, δEδp


class FakeError(sn.ErrorModel):
    def __init__(self,error=1,derivative_value=1,name=None):
        super().__init__()
        self.error=error
        self.derivative_value=1

    # Fake error function without parameters
    # forward always returns self.error
    # backward always returns array with same shape as input
    # and all values set to self.derivative_values
    def forward(self, y, y_true):
        self.set_cache(y.shape)
        return self.error

    def backward(self, δE:float):
        shape, = self.get_cache()
        return np.ones(shape)*self.derivative_value, {}