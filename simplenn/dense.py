from simplenn.layer import CommonLayer
import numpy as np
from .initializers import  Initializer,Zero,RandomNormal
from . import activations


class Bias(CommonLayer):
    '''
    The Bias layer outputs y = x+b, where b is a vector of parameters
    Input for forward:
    x: array of size (n,o)
    where $n$ is the batch size and $o$ is the number of both input and output features
    The number of columns of x, $o$, must match the size of $b$.

    '''
    def __init__(self,output_size:int,initializer:Initializer=Zero(),name=None):
        super().__init__(name=name)
        b = initializer.create( (output_size,))
        self.register_parameter("b", b)

    def forward_with_cache(self, x:np.ndarray):
        n,d = x.shape
        b = self.get_parameters()["b"]
        dout, = b.shape
        assert dout==d, f"#features of input ({d}) must match size of b ({dout})"
        ### COMPLETAR INICIO ###
        y = x + b
        ### COMPLETAR FIN ###
        cache = tuple()
        return y,cache

    def backward(self, δEδy:np.ndarray,cache):
        b = self.get_parameters()["b"]

        δEδx = np.zeros_like(δEδy)

        # Calculate derivative of error E with respect to input x
        # Hints:
        # δEδx = δEδy * δyδx = δEδy * [1,1,...,1] = δEδy
        ### COMPLETAR INICIO ###
        δEδx = δEδy
        ### COMPLETAR FIN ###

        # Calculate derivative of error E with respect to parameter b
        # Hints:
        # δEδb = δEδy * δyδb
        # δyδb = [1, 1, 1, ..., 1]
        n,d = δEδy.shape
        δEδb = np.zeros_like(b)
        for i in range(n):
            # Calculate derivative of error for a sample i (a single sample)
            # And accumulate to obtain δEδb
            ### COMPLETAR INICIO ###
            δEδb_i = δEδy[i,:] # * [1,1,1...,1]
            δEδb += δEδb_i
            ### COMPLETAR FIN ###

        return δEδx,{"b":δEδb}



class Linear(CommonLayer):

    '''
    The Linear layer outputs y = xw, where w is a matrix of parameters

    '''
    def __init__(self,input_size:int,output_size:int,initializer:Initializer=RandomNormal(),name=None):
        super().__init__(name=name)
        shape = (input_size,output_size)
        w = initializer.create(shape)
        self.register_parameter("w", w)


    def forward_with_cache(self, x:np.ndarray):
        n,d = x.shape
        # Retrieve w
        w = self.get_parameters()["w"]
        # check sizes
        din,dout = w.shape
        assert din==d, f"#features of input ({d}) must match first dimension of W ({din})"



        y = np.zeros((n,dout))
        # calculate output
        ### COMPLETAR INICIO ###
        y = x.dot(w)
        ### COMPLETAR FIN ###

        # add input to cache to calculate δEδw in backward step
        cache = (x,)
        return y,cache

    def backward(self,δEδy:np.ndarray,cache):
        # Retrieve input from cache to calculate δEδw
        x, = cache
        n = x.shape[0]

        # Retrieve w
        w = self.get_parameters()["w"]

        # Calculate derivative of error E with respect to input x

        # TODO move to exp

        δEδx = np.zeros_like(x)
        ### COMPLETAR INICIO ###
        # Per sample version
        # for i in range(n):
        #      δEδx[i,:] = np.dot(w, δEδy[i,:])

        # Vectorized version
        #δyδx = w.T
        δEδx =δEδy.dot(w.T)
        ### COMPLETAR FIN ###

        # Calculate derivative of error E with respect to parameter w
        δEδw = np.zeros_like(w)

        ### COMPLETAR INICIO ###
        # per sample version
        # for i in range(n):
        #      δEδw_i = np.outer(x[i,:], δEδy[i,:])
        #      δEδw += δEδw_i

        ## Vectorized version
        δEδw = x.T.dot(δEδy)
        ### COMPLETAR FIN ###

        return δEδx, {"w":δEδw}


activation_dict = {"id":activations.Identity,
                   "relu":activations.ReLU,
                   "tanh":activations.TanH,
                   "sigmoid":activations.Sigmoid,
                   "softmax":activations.Softmax,
                   }

class Dense(CommonLayer):
    '''
    A Dense layer simplifies the definition of networks by producing a common block
    that applies a linear, bias and activation function, in that order, to an input, ie
    y = activation(wx+b), where w and b are the parameters of the Linear and Bias layers,
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

    '''
    def __init__(self, input_size:int, output_size:int,activation_name:str=None,
                 linear_initializer:Initializer=RandomNormal(), bias_initializer:Initializer=Zero(), name=None):
        self.linear = Linear(input_size,output_size,initializer=linear_initializer)
        self.bias = Bias(output_size,initializer=bias_initializer)

        if activation_name is None:
            activation_name = "id"
        if activation_name in activation_dict:
            self.activation = activation_dict[activation_name]()
        else:
            raise ValueError(f"Unknown activation function {activation_name}. Available activations: {','.join(activation_dict.keys())}")

        super().__init__(name=name)
        # add activation name to Dense name
        self.name+=f"({activation_name})"

    def forward_with_cache(self, x:np.ndarray):
        # calculate and return activation(bias(linear(x)))

        ### COMPLETAR INICIO ###
        y_linear,cache_linear = self.linear.forward_with_cache(x)
        y_bias,cache_bias =self.bias.forward_with_cache(y_linear)
        y_activation,cache_activation= self.activation.forward_with_cache(y_bias)
        ### COMPLETAR FIN ###
        return y_activation, (cache_linear,cache_bias,cache_activation)

    def backward(self,δEδy:np.ndarray,cache):
        # Compute gradients for the parameters of the bias, linear and activation function
        # It is possible that the activation function does not have any parameters
        # (ie, δEδactivation = {})
        (cache_linear,cache_bias,cache_activation) = cache
        δEδbias,δEδlinear,δEδactivation={},{},{}
        ### COMPLETAR INICIO ###
        δEδx_activation,δEδactivation = self.activation.backward(δEδy,cache_activation)
        δEδx_bias,δEδbias =self.bias.backward(δEδx_activation,cache_bias)
        δEδx,δEδlinear =self.linear.backward(δEδx_bias,cache_linear)
        ### COMPLETAR FIN ###

        # combine gradients for parameters from dense, linear and activation layers
        δEδdense ={**δEδbias, **δEδlinear,**δEδactivation}
        return δEδx,δEδdense


    def get_parameters(self):
        # returns the combination of parameters of all models
        # assumes no Layer uses the same parameter names
        # ie: Linear has `w`, bias has `b` and activation
        # has a different parameter name (if it has any).
        p_linear = self.linear.get_parameters()
        p_bias = self.bias.get_parameters()
        p_activation = self.activation.get_parameters()
        p = {**p_linear, **p_bias,**p_activation}
        return p


