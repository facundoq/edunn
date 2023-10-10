# Additional material to help you implement optimizers:
# "An overview of gradient descent optimization algorithms" https://ruder.io/optimizing-gradient-descent/


from typing import Dict
import numpy as np
from .model import Model,Phase
from .model import  ParameterSet
import sys,abc
from tqdm.auto import tqdm

class Optimizer(abc.ABC):

    @abc.abstractmethod
    def optimize(self, model:Model, *args):
        pass


def all_equal(list:[]):
    return len(list) == 0 or list.count(list[0]) == len(list)
import random

def batch_arrays(batch_size:int,*arrays,shuffle=False):
    '''

    :param batch_size: size of batches
    :param arrays: variable number of numpy arrays
    :return: a generator that returns the arrays in batches
    '''

    sample_sizes = [a.shape[0] for a in arrays]
    assert all_equal(sample_sizes)
    batches=sample_sizes[0]//batch_size
    batch_list = list(range(batches))
    if shuffle:
        random.shuffle(batch_list)
    for i in batch_list:
        start = i*batch_size
        end = start+batch_size
        batch = [ a[start:end,] for a in arrays]
        yield tuple(batch)


class BatchedGradientOptimizer(Optimizer):

    def __init__(self,batch_size:int,epochs:int,shuffle=True):
        '''
        :param epochs: number of epochs to train the model. Each epoch is a complete iteration over the training set. The number of parameter updates is n //batch_size, where n is the number of samples of the dataset
        :param batch_size: Batch the dataset with batches of size `batch_size`, and perform an optimization step for each batch
        '''
        self.batch_size=batch_size
        self.epochs=epochs
        self.shuffle=shuffle

    def backpropagation(self,model:Model, x:np.ndarray, y_true:np.ndarray, error_layer:Model):
        # forward pass (model and error)
        y = model.forward(x)
        E = error_layer.forward(y_true, y)

        # backward pass (error and model)
        δEδy, _ = error_layer.backward(1)
        δEδx,δEδps = model.backward(δEδy)

        return δEδx,δEδps,E

    def optimize(self, model:Model, x:np.ndarray, y:np.ndarray, error_layer:Model, verbose=True):
        '''
        Fit a model to a dataset.
        :param model: the Model to optimize
        :param x: dataset inputs
        :param y: dataset outputs
        :param error_layer: To be applied to the output of the last layer
        :return:
        '''
        n = x.shape[0]
        batches = n // self.batch_size
        history = []
        model.set_phase(Phase.Training)
        bar = tqdm(range(self.epochs),desc=f"optim. {model.name}",file=sys.stdout,disable=not verbose)
        for epoch in bar:
            epoch_error=0
            for i,(x_batch,y_batch) in enumerate(batch_arrays(self.batch_size,x,y,shuffle=self.shuffle)):
                δEδx, δEδps, batch_error = self.backpropagation(model, x, y, error_layer)
                self.optimize_batch(model,δEδps,epoch,i)
                epoch_error+=batch_error
            epoch_error/=batches
            history.append(epoch_error)
            bar.set_postfix_str(f"{error_layer.name}: {epoch_error:.5f}")

        return np.array(history)

    @abc.abstractmethod
    def optimize_batch(self, model:Model, x:np.ndarray, y:np.ndarray, error_layer:Model, epoch:int):
        pass

class GradientDescent(BatchedGradientOptimizer):

    def __init__(self,batch_size:int,epochs:int,lr:float=0.1,shuffle=True):
        super().__init__(batch_size,epochs,shuffle)
        self.lr=lr

    def optimize_batch(self, model:Model, δEδps:ParameterSet, epoch:int,iteration:int):

        # Update parameters
        parameters = model.get_parameters()
        for parameter_name,δEδp in δEδps.items():
                p = parameters[parameter_name]
                # use p[:] so that updates are in-place
                # instead of creating a new variable
                ### YOUR IMPLEMENTATION START  ###
                p[:] = p - self.lr * δEδp
                ### YOUR IMPLEMENTATION END  ###

class MomentumGD(BatchedGradientOptimizer):

    def __init__(self,batch_size:int,epochs:int,lr:float=0.1,gamma=0.9,shuffle=True):
        super().__init__(batch_size,epochs,shuffle)
        self.lr=lr
        self.gamma=gamma
        self.first=True
        self.v={}

    def optimize_batch(self, model:Model, δEδps:ParameterSet, epoch:int,iteration:int):
        if self.first:
            self.first=False
            for k,p in model.get_parameters().items():
                self.v[k]=np.zeros_like(p)

        # Update parameters
        parameters = model.get_parameters()
        for k,δEδp in δEδps.items():
            # K = parameter name
            p = parameters[k]
            v = self.v[k]
            # use p[:] and v[:] so that updates are in-place
            # instead of creating a new variable
            ### YOUR IMPLEMENTATION START  ###
            v[:] = self.gamma * v + self.lr * δEδp
            p[:] = p - v
            ### YOUR IMPLEMENTATION END  ###


class NesterovMomentumGD(BatchedGradientOptimizer):

    def __init__(self,batch_size:int,epochs:int,lr:float=0.1,gamma=0.9,shuffle=True):
        super().__init__(batch_size,epochs,shuffle)
        self.lr=lr
        self.gamma=gamma
        self.first=True
        self.v={}

    def optimize_batch(self, model:Model, δEδps:ParameterSet, epoch:int,iteration:int):
        if self.first:
            self.first=False
            for k,p in model.get_parameters().items():
                self.v[k]=np.zeros_like(p)

        # Update parameters
        parameters = model.get_parameters()
        for k,δEδp in δEδps.items():
            # K = parameter name
            p = parameters[k]
            v = self.v[k]
            # use p[:] so that updates are in-place
            # instead of creating a new variable
            ### YOUR IMPLEMENTATION START  ###
            v[:] = self.gamma * v + self.lr * δEδp
            p[:] = p - (self.gamma*v+self.lr*δEδp)
            ### YOUR IMPLEMENTATION END  ###


class SignGD(BatchedGradientOptimizer):

    def __init__(self,batch_size:int,epochs:int,lr:float=0.1,eps=1e-8,shuffle=True):
        super().__init__(batch_size,epochs,shuffle)
        self.eps=eps
        self.lr=lr

    def optimize_batch(self, model:Model, δEδps:ParameterSet, epoch:int,iteration:int):

        # Update parameters
        parameters = model.get_parameters()
        for parameter_name,δEδp in δEδps.items():
                p = parameters[parameter_name]
                # use p[:] so that updates are in-place
                # instead of creating a new variable
                ### YOUR IMPLEMENTATION START  ###
                denom = np.sqrt(δEδp**2+self.eps)
                p[:] = p - self.lr * (δEδp/denom)
                ### YOUR IMPLEMENTATION END  ###

