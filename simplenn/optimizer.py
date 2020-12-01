from typing import Dict
import numpy as np
import abc

class Optimizer(abc.ABC):

    @abc.abstractmethod
    def optimize(self,weights:[Dict[str,np.ndarray]],gradients:[Dict[str,np.ndarray]],names:[str]):
        pass



class RandomOptimizer(Optimizer):
    '''
    Random Optimizer
    Takes a step in a random direction
    '''
    def __init__(self,lr=0.001):
        self.lr=lr

    def optimize(self,weights:[Dict[str,np.ndarray]],gradients:[Dict[str,np.ndarray]],names:[str]):
        for (layer_weights,layer_gradients) in zip(weights,gradients):
            for parameter_name, w in layer_weights.items():
                g = layer_gradients[parameter_name] # ignored
                # use w[:] so that updates are in-place
                # instead of creating a new variable
                w[:] = w + (np.random.random_sample(w.shape)-0.5)*self.lr

class GradientDescent(Optimizer):
    '''
    GradientDescent optimizer
    Uses the gradient to update weights with the rule
    w = w - lr * g
    Where g is the gradient and lr is the learning rate
    '''
    def __init__(self,lr=0.001):
        self.lr=lr

    def optimize(self,weights:[Dict[str,np.ndarray]],gradients:[Dict[str,np.ndarray]],names:[str]):
        for (layer_weights,layer_gradients,name) in zip(weights,gradients,names):
            for parameter_name,w in layer_weights.items():
                g = layer_gradients[parameter_name]
                # use w[:] so that updates are in-place
                # instead of creating a new variable
                delta =self.lr * g
                w[:] = w - delta