import edunn as nn
import numpy as np
from typing import Tuple,Dict,Callable

ParameterSet = Dict[str,np.ndarray]

def set_parameters_inplace(l:nn.Model, parameters:ParameterSet):
    old_params = l.get_parameters()
    for k, v in old_params.items():
        l.get_parameters()[k][:]=parameters[k]

def common_layer_to_function(l:nn.ModelWithParameters):

    def f(inputs:Dict[str,np.ndarray]):
        old_params = l.get_parameters().copy()
        set_parameters_inplace(l,inputs)
        y = l.forward(inputs["x"])

        set_parameters_inplace(l,old_params)
        return y

    def df(inputs:ParameterSet,δEδy:np.ndarray):
        old_params = l.get_parameters().copy()
        set_parameters_inplace(l,inputs)
        y=l.forward(inputs["x"])
        δEδx,δEδp = l.backward(δEδy)
        set_parameters_inplace(l,old_params)
        return {"x":δEδx,**δEδp}
    parameter_shapes = {k:v.shape for k,v in l.get_parameters().items()}
    return f,df,parameter_shapes

def error_layer_to_function(l:nn.ErrorModel):

    def f(inputs:Dict[str,np.ndarray]):
        old_params = l.get_parameters().copy()
        set_parameters_inplace(l,inputs)
        y = l.forward(inputs["y_true"], inputs["y"])
        set_parameters_inplace(l,old_params)
        return y

    def df(inputs:Dict[str,np.ndarray]):
        old_params = l.get_parameters().copy()
        set_parameters_inplace(l,inputs)
        y = l.forward(inputs["y_true"], inputs["y"])
        δEδy,δEδp = l.backward(np.ones_like(y))
        set_parameters_inplace(l,old_params)
        return {"y":δEδy,**δEδp}
    parameter_shapes = {k:v.shape for k,v in l.get_parameters().items()}

    return f,df,parameter_shapes
