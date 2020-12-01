from simplenn.layer import Layer
import numpy as np
from .initializers import  Initializer,Zero,RandomUniform,XavierUniform



class Bias(Layer):
    '''
    The Bias layer outputs y = x+b, where b is a vector of parameters
    Its derivatives wrt to its input (x) and parameters (b) are:
    δyδx = [1,1,1,...,1]
    δyδb = [1,1,1,...1]
    '''
    def __init__(self,output_size:int,initializer:Initializer=Zero(),name=None):
        super().__init__(name=name)
        b = initializer.create( (output_size,))
        self.register_parameter("b", b)

    def forward(self,x:np.ndarray):
        n,d = x.shape
        b = self.get_parameters()["b"]
        dout, = b.shape
        assert dout==d, f"#features of input ({d}) must match size of b ({dout})"
        ### COMPLETAR INICIO ###
        y = x + b
        ### COMPLETAR FIN ###
        return y

    def backward(self, δEδy:np.ndarray):
        b = self.get_parameters()["b"]
        # Calculate derivative of error E with respect to input x
        # δEδx = δEδy * δyδx = δEδy * [1,1,...,1] = δEδy
        δEδx =δEδy

        # Calculate derivative of error E with respect to parameter b
        # δEδb = δEδy * δyδb
        # δyδb = [1, 1, 1, ..., 1]
        n,d = δEδy.shape
        δEδb = np.zeros_like(b)
        for i in range(n):
            ### COMPLETAR INICIO ###
            δEδb_i = δEδy[i,:] # * [1,1,1...,1]
            δEδb += δEδb_i
            ### COMPLETAR FIN ###

        return δEδx,{"b":δEδb}



class Linear(Layer):

    def __init__(self,input_size:int,output_size:int,initializer:Initializer=RandomUniform(),name=None):
        super().__init__(name=name)
        shape = (input_size,output_size)
        w = initializer.create(shape)
        self.register_parameter("w", w)


    def forward(self,x:np.ndarray):
        n,d = x.shape
        # Retrieve w
        w = self.get_parameters()["w"]
        # check sizes
        din,dout = w.shape
        assert din==d, f"#features of input ({d}) must match first dimension of W ({din})"

        # calculate output
        ### COMPLETAR INICIO ###
        y = x.dot(w)
        ### COMPLETAR FIN ###

        # add input to cache to calculate δEδw in backward step
        self.set_cache(x)
        return y

    def backward(self,δEδy:np.ndarray):
        # Retrieve input from cache to calculate δEδw
        x, = self.cache
        # Retrieve w
        w = self.get_parameters()["w"]

        # Calculate derivative of error E with respect to input x

        # TODO move to exp
        # δEδx = δEδy * δyδx = δEδy * w
        # N = number of samples
        # O = output dim
        # I = input dim
        # δEδy.shape = NxO
        # w.shape = IxO
        # δEδx.shape = NxI
        # → δyδx.shape = OxI

        ### COMPLETAR INICIO ###
        δyδx = w.T
        δEδx =δEδy.dot(δyδx)
        ### COMPLETAR FIN ###

        # Calculate derivative of error E with respect to parameter w
        # δEδw = δEδy * δyδw = δEδy * x
        δEδw = np.zeros_like(w)
        ## Vectorized
        δEδw = x.T.dot(δEδy)

        ## TODO non-vectorized version
        n,d = δEδy.shape
        # for i in range(n):
        ### COMPLETAR INICIO ###
        #     print(x[i,:].shape,δEδy[i,:].shape)
        #     δEδw_i = δEδy[i,:].dot(x[i,:])
        #     δEδw += δEδw_i

        ### COMPLETAR FIN ###


        return δEδx, {"w":δEδw}




class Dense(Layer):
    def __init__(self,input_size:int,output_size:int,
                 linear_initializer:Initializer=XavierUniform(),bias_initializer:Initializer=Zero(),name=None):
        self.linear = Linear(input_size,output_size,initializer=linear_initializer)
        self.bias = Bias(output_size,initializer=bias_initializer)
        super().__init__(name=name)

    def forward(self,x:np.ndarray):
        return self.bias.forward(self.linear.forward(x))

    def backward(self,δEδy:np.ndarray):

        ### COMPLETAR INICIO ###
        δEδx,δEδbias =self.bias.backward(δEδy)
        δEδx,δEδlinear =self.linear.backward(δEδx)
        ### COMPLETAR FIN ###
        δEδdense ={**δEδbias, **δEδlinear}
        return δEδx,δEδdense

    def reset(self):
        self.linear.reset()
        self.bias.reset()

    def get_parameters(self):
        p_linear = self.linear.get_parameters()
        p_bias = self.bias.get_parameters()
        p = {**p_linear, **p_bias}
        return p


