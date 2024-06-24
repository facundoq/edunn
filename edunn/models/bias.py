from edunn.model import ModelWithParameters
import numpy as np
from edunn.initializers import Initializer, Zero
import edunn


class Bias(ModelWithParameters):
    """
    The Bias layer outputs y = x+b, where b is a vector of parameters
    Input for forward:
    x: array of size (n,o)
    where $n$ is the batch size and $o$ is the number of both input and output features
    The number of columns of x, $o$, must match the size of $b$.

    """

    def __init__(self, output_size: int, initializer: edunn.initializers.Initializer = None, name=None):
        super().__init__(name=name)
        if initializer is None:
            initializer = edunn.initializers.Zero()
        b = initializer.create((output_size,))
        self.register_parameter("b", b)

    def forward(self, x: np.ndarray):
        n, d = x.shape
        b = self.get_parameters()["b"]
        dout, = b.shape
        assert dout == d, f"#features of input ({d}) must match size of b ({dout})"
        y = np.zeros_like(x)
        ### YOUR IMPLEMENTATION START  ###

        y = x + b

        ### YOUR IMPLEMENTATION END  ###
        return y

    def backward(self, dE_dy: np.ndarray):
        b = self.get_parameters()["b"]
        dE_dx = np.zeros_like(dE_dy)

        # Calculate derivative of error E with respect to input x
        # Hints:
        # dE_dx = dE_dy * dy_dx = dE_dy * [1,1,...,1] = dE_dy
        ### YOUR IMPLEMENTATION START  ###
        dE_dx = dE_dy
        ### YOUR IMPLEMENTATION END  ###

        # Calculate derivative of error E with respect to parameter b

        # Hints:
        # dE_db = dE_dy * dy_db
        # dy_db = [1, 1, 1, ..., 1]
        n, d = dE_dy.shape
        dE_db = np.zeros_like(b)
        for i in range(n):
            # Calculate derivative of error for a sample i (a single sample)
            # And accumulate to obtain dE_db
            ### YOUR IMPLEMENTATION START  ###
            dE_db_i = dE_dy[i, :]  # * [1,1,1...,1]
            dE_db += dE_db_i
            ### YOUR IMPLEMENTATION END  ###

        return dE_dx, {"b": dE_db}
