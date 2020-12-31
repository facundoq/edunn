import numpy as np
import simplenn as sn
from typing import Tuple,Dict


def numerical_gradient(f,x:np.ndarray,δEδy:np.ndarray, h=1e-5):
    ''' Calculates the numerical gradient of E wrt x'''
    ''' E is assumed to be a scalar, so that δEδy has size equal to y'''
    def indices_generator(x):
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            # evaluate function at x+h
            ix = it.multi_index
            yield ix
            it.iternext()
    h2 = 2*h
    δEδx = np.zeros_like(x)
    for i in indices_generator(x):
        oldval = x[i]
        x[i] = oldval + h # increment by h
        fxph = f(x) # evaluate f(x + h)
        x[i] = oldval - h # increment by h
        fxmh = f(x) # evaluate f(x - h)
        x[i] = oldval # reset

        δyδxi = (fxph - fxmh)/h2
        δE = (δyδxi*δEδy).sum()
        δEδx[i] = δE
    return δEδx



def f(layer:sn.Model, x:np.ndarray): return layer.forward(x)

def df(layer:sn.Model, x:np.ndarray, δEδy:np.ndarray):

    layer.reset()
    y = f(layer,x)
    δEδx,δEδp = layer.backward(δEδy)
    return δEδx,δEδp,y



def layer_to_functions(l:sn.model):
    fx = lambda x:f(l, x)

    def fp(l:sn.Model, x:np.ndarray, k:str, p:np.ndarray):
        old_p=l.get_parameters()[k]
        l.get_parameters()[k][:]=p
        y = l.forward(x)
        l.get_parameters()[k][:]=old_p
        return y

    fps={}
    for k in l.get_parameters():
        fps[k] =lambda p,x:fp(l, x, k, p)

    return fx,fps

def replace_params(l:sn.Model, new_p:Dict[str, np.ndarray]):
    current_p = l.get_parameters()
    for k in current_p:
        current_p[k][:]=new_p[l]