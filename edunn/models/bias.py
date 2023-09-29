from edunn.model import ModelWithParameters
import numpy as np
from edunn.initializers import Initializer, Zero
import edunn

class Bias(ModelWithParameters):
    '''
    The Bias layer outputs y = x+b, where b is a vector of parameters
    Input for forward:
    x: array of size (n,o)
    where $n$ is the batch size and $o$ is the number of both input and output features
    The number of columns of x, $o$, must match the size of $b$.

    '''

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

    def backward(self, δEδy: np.ndarray):
        b = self.get_parameters()["b"]
        δEδx = np.zeros_like(δEδy)

        # Calculate derivative of error E with respect to input x
        # Hints:
        # δEδx = δEδy * δyδx = δEδy * [1,1,...,1] = δEδy
        ### YOUR IMPLEMENTATION START  ###
        δEδx = δEδy
        ### YOUR IMPLEMENTATION END  ###

        # Calculate derivative of error E with respect to parameter b

        # Hints:
        # δEδb = δEδy * δyδb
        # δyδb = [1, 1, 1, ..., 1]
        n, d = δEδy.shape
        δEδb = np.zeros_like(b)
        for i in range(n):
            # Calculate derivative of error for a sample i (a single sample)
            # And accumulate to obtain δEδb
            ### YOUR IMPLEMENTATION START  ###
            δEδb_i = δEδy[i, :]  # * [1,1,1...,1]
            δEδb += δEδb_i
            ### YOUR IMPLEMENTATION END  ###

        return δEδx, {"b": δEδb}
