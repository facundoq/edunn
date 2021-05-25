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


    def set_phase(self,phase:Phase):
        for l in self.layers:
            l.set_phase(phase)

    def backward(self,δEδy:np.ndarray):
        '''

        :param x: inputs
        :param y: expected output
        :return: gradients for every layer, prediction for inputs and error
        '''

        gradients={}
        for layer in reversed(self.layers):
            δEδy,δEδp = layer.backward(δEδy)
            if not layer.frozen:
                for k,v in δEδp.items():
                    new_name=self.generate_parameter_name(layer,k)
                    gradients[new_name]=v

        return δEδy, gradients

    def generate_parameter_name(self,l:Model,parameter_name:str):
        return f"{l.name}({parameter_name})"

    def get_parameters(self):
        parameters={}
        for l in self.layers:
            for k,v in l.get_parameters().items():
                new_name=self.generate_parameter_name(l,k)
                parameters[new_name]=v
        return parameters

    def summary(self)->str:
        '''
        :return: a summary of the models of the model and their parameters
        '''
        separator = "-------------------------------"
        result=f"{separator}\n"
        parameters=0
        result+=f"Model {self.name}:\n"
        for l in self.layers:
            layer_parameters= l.n_parameters()
            parameters+=layer_parameters
            l_summary=f"{l.name} → params: {layer_parameters}"
            result+=l_summary+"\n"
        result+=f"Total parameters: {parameters}\n"
        result+=f"{separator}\n"
        return result
