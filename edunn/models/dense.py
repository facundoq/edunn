import numpy as np
from ..model import ModelWithParameters

from ..initializers import Initializer, Zero, RandomNormal

from .linear import Linear
from . import activations
from .bias import Bias

activation_dict = {"id": activations.Identity,
                   "relu": activations.ReLU,
                   "tanh": activations.TanH,
                   "sigmoid": activations.Sigmoid,
                   "softmax": activations.Softmax,
                   }


class Dense(ModelWithParameters):
    """
    A Dense layer simplifies the definition of networks by producing a common block
    that applies a linear, bias and activation function, in that order, to an input, ie
    y = activation(wx+b), where w and b are the parameters of the Linear and Bias models,
    and activation is the function of an activation Layer.

    Therefore, a defining a Dense layer such as:

    ```
    [...
    Dense(input_size,output_size,activation_name="relu")
    ]
    ```

    Is equivalent to:

    ```[...
    Linear(input_size,output_size),
    Bias(output_size)
    ReLu(),...]
    ```

    By default, no activation is used (actually, the Identity activation is used, which
    is equivalent). Implemented activations:
    * id
    * relu
    * tanh
    * sigmoid
    * softmax
    """

    def __init__(self, input_size: int, output_size: int, activation_name: str = None,
                 linear_initializer: Initializer = None, bias_initializer: Initializer = None, name=None):
        self.linear = Linear(input_size, output_size, initializer=linear_initializer)
        self.bias = Bias(output_size, initializer=bias_initializer)

        if activation_name is None:
            activation_name = "id"
        if activation_name in activation_dict:
            self.activation = activation_dict[activation_name]()
        else:
            raise ValueError(
                f"Unknown activation function {activation_name}. Available activations: {','.join(activation_dict.keys())}")

        super().__init__(name=name)
        # add activation name to Dense name
        self.name += f"({activation_name})"

    def forward(self, x: np.ndarray):
        # calculate and return activation(bias(linear(x)))
        y_activation = None
        """ YOUR IMPLEMENTATION START """
        y_linear = self.linear.forward(x)
        y_bias = self.bias.forward(y_linear)
        y_activation = self.activation.forward(y_bias)
        """ YOUR IMPLEMENTATION END """
        return y_activation

    def backward(self, dE_dy: np.ndarray):
        # Compute gradients for the parameters of the bias, linear and activation function
        # It is possible that the activation function does not have any parameters
        # (ie, dE_dactivation = {})
        dE_dbias, dE_dlinear, dE_dactivation = {}, {}, {}
        dE_dx = None
        """ YOUR IMPLEMENTATION START """
        dE_dx_activation, dE_dactivation = self.activation.backward(dE_dy)
        dE_dx_bias, dE_dbias = self.bias.backward(dE_dx_activation)
        dE_dx, dE_dlinear = self.linear.backward(dE_dx_bias)
        """ YOUR IMPLEMENTATION END """

        # combine gradients for parameters from dense, linear and activation models
        dE_ddense = {**dE_dbias, **dE_dlinear, **dE_dactivation}
        return dE_dx, dE_ddense

    def get_parameters(self):
        # returns the combination of parameters of all models
        # assumes no Layer uses the same parameter names
        # ie: Linear has `w`, bias has `b` and activation
        # has a different parameter name (if it has any).
        p_linear = self.linear.get_parameters()
        p_bias = self.bias.get_parameters()
        p_activation = self.activation.get_parameters()
        p = {**p_linear, **p_bias, **p_activation}
        return p
