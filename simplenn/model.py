import numpy as np
from tqdm.auto import tqdm
from simplenn.layer import Layer,ErrorLayer,Phase
import abc
from simplenn.optimizer import Optimizer
import random


class Model(abc.ABC):
    '''
    Abstract class model
    '''
    @abc.abstractmethod
    def predict(self,*args):
        pass

    @abc.abstractmethod
    def fit(self,*args):
        pass



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
        batch = [ a[start:end,:] for a in arrays]
        yield tuple(batch)



def all_equal(list:[]):
    return len(list) == 0 or list.count(list[0]) == len(list)

class Sequential(Model):
    '''
        Models a neural network with a sequential (ie, linear) topology
        This network receives as input a single vector x
        And outputs a single vector y
    '''


    def __init__(self, layers:[Layer]):
        '''

        :param layers: List of layers, in order

        '''
        self.layers:[Layer]=layers

    def predict(self,x:np.ndarray):
        '''

        :param x: input to model
        :return: output of model with x as input
        '''
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward_with_cache(self,x:np.ndarray):
        '''

        :param x: input to model
        :return: output of model with x as input
                 cache of each layer
        '''
        caches=[]
        for layer in self.layers:
            x,cache = layer.forward_with_cache(x)
            caches.append(cache)
        return x,caches

    def fit(self, x:np.ndarray, y:np.ndarray, error_layer:ErrorLayer, epochs:int, batch_size:int, optimizer:Optimizer):
        '''
        Fit a model to a dataset.
        :param x: dataset inputs
        :param y: dataset outputs
        :param error_layer: To be applied to the output of the last layer
        :param epochs: number of epochs to train the model. Each epoch is a complete iteration over the training set. The number of parameter updates is n //batch_size, where n is the number of samples of the dataset
        :param batch_size: Batch the dataset with batches of size `batch_size`.
        :param algorithm optimizer: used to modify the parameters of the model
        :return:
        '''
        n = x.shape[0]
        batches = n // batch_size
        history = []
        bar = tqdm(range(epochs),desc="fit")
        for epoch in bar:
            epoch_error=0
            for x_batch,y_batch in batch_arrays(batch_size,x,y):
                batch_error=self.fit_batch(x_batch,y_batch,error_layer,optimizer)
                epoch_error+=batch_error
            epoch_error/=batches
            history.append(epoch_error)
            bar.set_postfix_str(f"error: {epoch_error:.5f}")

        return np.array(history)

    def set_phase(self,phase:Phase):
        for l in self.layers:
            l.set_phase(phase)

    def backward(self,x:np.ndarray,y:np.ndarray,error_layer:ErrorLayer,average_batch=True):
        '''

        :param x: inputs
        :param y: expected output
        :return: gradients for every layer, prediction for inputs and error
        '''
        n = x.shape[0]
        y_pred,caches = self.forward_with_cache(x)
        E,E_cache = error_layer.forward_with_cache(y, y_pred)
        δEδy = error_layer.backward(E_cache)
        gradients=[]
        for layer,cache in reversed(list(zip(self.layers,caches))):
            δEδy,δEδp = layer.backward(δEδy,cache)
            if average_batch:
                # divide gradients by batch size to obtain average gradients
                # so that magnitudes are independent of batch size
                for k,v in δEδp.items():
                    v[:]/=n
            if not layer.frozen:
                # insert beginning
                gradients.insert(0,δEδp)
        return gradients, y_pred, E

    def reset_layers(self):
        for l in self.layers:
            l.reset()

    def fit_batch(self,x:np.ndarray,y:np.ndarray,error_layer:ErrorLayer,optimizer:Optimizer):
        '''
        Fit model on a batch of samples
        :param x: input samples
        :param y: target samples
        :param error_layer: Error function to optimize
        :param optimizer: algorithm used to modify the parameters of the model
        :return: error of the network on this batch
        '''
        self.set_phase(Phase.Training)
        #self.reset_layers()
        gradients, y_pred,error = self.backward(x,y,error_layer)
        # list of parameters for each layer
        parameters = [l.get_parameters() for l in self.layers if not l.frozen]
        names = [l.name for l in self.layers]
        optimizer.optimize(parameters,gradients,names)
        return error


    def summary(self)->str:
        '''
        :return: a summary of the layers of the model and their parameters
        '''
        separator = "-------------------------------"
        result=f"{separator}\n"
        parameters=0
        for l in self.layers:
            layer_parameters= l.n_parameters()
            parameters+=layer_parameters
            l_summary=f"{l.name} → params: {layer_parameters}"
            result+=l_summary+"\n"
        result+=f"Total parameters: {parameters}\n"
        result+=f"{separator}\n"
        return result



