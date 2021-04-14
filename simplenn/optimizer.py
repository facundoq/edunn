from typing import Dict
import numpy as np
from .model import Model
from .model import ErrorModel, ParameterSet
import sys,abc
from tqdm.auto import tqdm

class Optimizer(abc.ABC):

    @abc.abstractmethod
    def optimize(self, model:Model, *args):
        pass




def all_equal(list:[]):
    return len(list) == 0 or list.count(list[0]) == len(list)
import random

def batch_arrays(batch_size:int,*arrays,shuffle=True):
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


class BatchedOptimizer(Optimizer):

    def __init__(self,batch_size:int,epochs:int):
        '''
        :param epochs: number of epochs to train the model. Each epoch is a complete iteration over the training set. The number of parameter updates is n //batch_size, where n is the number of samples of the dataset
        :param batch_size: Batch the dataset with batches of size `batch_size`, and perform an optimization step for each batch
        '''
        self.batch_size=batch_size
        self.epochs=epochs


    def optimize(self, model:Model, x:np.ndarray, y:np.ndarray, error_layer:ErrorModel, verbose=True):
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
        bar = tqdm(range(self.epochs),desc="fit",file=sys.stdout,disable=not verbose)
        for epoch in bar:
            epoch_error=0
            for x_batch,y_batch in batch_arrays(self.batch_size,x,y):
                batch_error=self.optimize_batch(model,x_batch,y_batch,error_layer,epoch)
                epoch_error+=batch_error
            epoch_error/=batches
            history.append(epoch_error)
            bar.set_postfix_str(f"{error_layer.name}: {epoch_error:.5f}")

        return np.array(history)

    @abc.abstractmethod
    def optimize_batch(self, model:Model, x:np.ndarray, y:np.ndarray, error_layer:ErrorModel, epoch:int):
        pass

class StochasticGradientDescent(BatchedOptimizer):

    def __init__(self,batch_size:int,epochs:int,lr:float=0.1):
        super().__init__(batch_size,epochs)
        self.lr=lr

    def optimize_batch(self, model:Model, x:np.ndarray, y_true:np.ndarray, error_layer:ErrorModel, epoch:int):
        y = model.forward(x)

        E = error_layer.forward(y_true, y)
        δEδy,_ = error_layer.backward(1)

        gradients = model.backward(δEδy)

        parameters = model.get_parameters()
        # print(parameters.keys())
        for parameter_name,δEδp in gradients.items():
                p = parameters[parameter_name]
                # print(parameter_name,p,δEδp)
                # use p[:] so that updates are in-place
                # instead of creating a new variable
                p[:] = p - self.lr * δEδp
        return E




# class RandomOptimizer(Optimizer):
#     '''
#     Random Optimizer
#     Takes a step in a random direction
#     '''
#     def __init__(self,lr=0.001):
#         self.lr=lr
#
#     def optimize(self,weights:[Dict[str,np.ndarray]],gradients:[Dict[str,np.ndarray]],names:[str]):
#         for (layer_weights,layer_gradients) in zip(weights,gradients):
#             for parameter_name, w in layer_weights.items():
#                 g = layer_gradients[parameter_name] # ignored
#                 # use w[:] so that updates are in-place
#                 # instead of creating a new variable
#                 w[:] = w + (np.random.random_sample(w.shape)-0.5)*self.lr