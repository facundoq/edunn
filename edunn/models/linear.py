from edunn.model import ModelWithParameters
import numpy as np
from edunn.initializers import Initializer, RandomNormal


class Linear(ModelWithParameters):
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
        """YOUR IMPLEMENTATION START"""
        y = x.dot(w)
        """YOUR IMPLEMENTATION END"""

        # add input to cache to calculate δEδw in backward step
        self.set_cache(x)
        return y

    def backward(self, δEδy: np.ndarray):
        # Retrieve input from cache to calculate δEδw
        (x,) = self.get_cache()
        n = x.shape[0]

        # Retrieve w
        w = self.get_parameters()["w"]

        # Calculate derivative of error E with respect to input x
        δEδx = np.zeros_like(x)
        """YOUR IMPLEMENTATION START"""

        # Per sample version
        # for i in range(n):
        #      δEδx[i,:] = np.dot(w, δEδy[i,:])

        # Vectorized version
        # δyδx = w.T
        δEδx = δEδy.dot(w.T)

        """YOUR IMPLEMENTATION END"""

        # Calculate derivative of error E with respect to parameter w
        δEδw = np.zeros_like(w)

        """YOUR IMPLEMENTATION START"""

        # Per sample version
        # for i in range(n):
        #      δEδw_i = np.outer(x[i,:], δEδy[i,:])
        #      δEδw += δEδw_i

        # Vectorized version
        δEδw = x.T.dot(δEδy)

        """YOUR IMPLEMENTATION END"""

        return δEδx, {"w": δEδw}
