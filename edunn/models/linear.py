from edunn.model import Model
import numpy as np
from edunn.initializers import Initializer, RandomNormal


class Linear(Model):
    """
    The Linear layer outputs y = xw, where w is a matrix of parameters
    """

    def __init__(self, input_size: int, output_size: int, initializer: Initializer = None, name=None):
        super().__init__(name=name)
        if initializer is None:
            initializer = RandomNormal()
        shape = (input_size, output_size)
        w = initializer.create(shape)
        self.register_parameter("w", w)

    def forward(self, x: np.ndarray):
        n, d = x.shape
        # Retrieve w
        w = self.get_parameters()["w"]
        # check sizes
        din, dout = w.shape
        assert din == d, f"#features of input ({d}) must match first dimension of W ({din})"

        y = np.zeros((n, dout))
        # calculate output
        """ YOUR IMPLEMENTATION START """
        y = x.dot(w)
        """ YOUR IMPLEMENTATION END """

        # add input to cache to calculate dE_dw in backward step
        self.set_cache(x)
        return y

    def backward(self, dE_dy: np.ndarray):
        # Retrieve input from cache to calculate dE_dw
        x, = self.get_cache()
        n = x.shape[0]

        # Retrieve w
        w = self.get_parameters()["w"]

        # Calculate derivative of error E with respect to input x
        dE_dx = np.zeros_like(x)
        """ YOUR IMPLEMENTATION START """

        # Per sample version
        # for i in range(n):
        #      dE_dx[i,:] = np.dot(w, dE_dy[i,:])

        # Vectorized version
        # dy_dx = w.T
        dE_dx = dE_dy.dot(w.T)

        """ YOUR IMPLEMENTATION END """

        # Calculate derivative of error E with respect to parameter w
        dE_dw = np.zeros_like(w)

        """ YOUR IMPLEMENTATION START """

        # Per sample version
        # for i in range(n):
        #      dE_dw_i = np.outer(x[i,:], dE_dy[i,:])
        #      dE_dw += dE_dw_i

        ## Vectorized version
        dE_dw = x.T.dot(dE_dy)

        """ YOUR IMPLEMENTATION END """

        return dE_dx, {"w": dE_dw}
