import numpy as np
from simplenn.model import Model,Phase


class Sequential(Model):
    '''
        Models a neural network with a sequential (ie, linear) topology
        This network receives as input a single vector x
        And outputs a single vector y
    '''

    def __init__(self, layers:[Model], name=None):
        '''

        :param layers: List of models, in order

        '''
        super().__init__(name)
        self.layers:[Model]=layers
        layer_names=[l.name for l in layers]
        layer_names_set=set(layer_names)
        assert len(layer_names)==len(layer_names_set), f"Layer names must be unique, found: {layer_names}"

    def forward(self, x:np.ndarray):
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



    def set_phase(self,phase:Phase):
        for l in self.layers:
            l.set_phase(phase)

    def backward(self,δEδy:np.ndarray,caches:[]):
        '''

        :param x: inputs
        :param y: expected output
        :return: gradients for every layer, prediction for inputs and error
        '''

        gradients={}
        for layer,cache in reversed(list(zip(self.layers,caches))):
            δEδy,δEδp = layer.backward(δEδy,cache)
            if not layer.frozen:
                for k,v in δEδp.items():
                    gradients[layer.name+k]=v
            return gradients

    def get_parameters(self):
        parameters={}
        for l in self.layers:
            for k,v in l.get_parameters().items():
                parameters[l.name+k]=v

    def summary(self)->str:
        '''
        :return: a summary of the models of the model and their parameters
        '''
        separator = "-------------------------------"
        result=f"{separator}\n"
        parameters=0
        print(f"Model {self.name}:")
        for l in self.layers:
            layer_parameters= l.n_parameters()
            parameters+=layer_parameters
            l_summary=f"{l.name} → params: {layer_parameters}"
            result+=l_summary+"\n"
        result+=f"Total parameters: {parameters}\n"
        result+=f"{separator}\n"
        return result
