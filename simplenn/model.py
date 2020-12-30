from abc import ABC,abstractmethod
import numpy as np
from enum import Enum
class_counter = {}
from typing import Tuple,Dict

class Phase(Enum):
    Training = "Training"
    Test = "Test"
ParameterSet = Dict[str,np.ndarray]
Cache = Tuple

class Model(ABC):
    '''

    A Model can perform forward and backward operations
    Each model has a unique name. Also, it has a set of parameters that can be updated.
    The forward operation takes an input (x) and generates an output (y)
    The backward takes an error gradient for the output (δEδy) and generates error gradients for the input (δEδx) and parameters (δEδp)
    The backward operation must be defined explicitly and is not automatically derived from the forward.
    The model can be frozen, so that its parameters are not updated/optimized.
    '''
    def __init__(self,name=None):
        if name is None:
            class_name= self.__class__.__name__
            count=class_counter.get(class_name, 0)
            name =f"{class_name}_{count}"
            class_counter[class_name]= count + 1
        self.phase=Phase.Training

        self.name=name
        self.frozen=False


    def set_phase(self,phase:Phase):
        '''
        Indicate to the model the current Phase (training/test/other) the model is in.
        This can be used to code different behaviors in training and testing phases.
        :param phase: phase to set
        '''
        self.phase = phase

    def n_parameters(self)->int:
        parameters=0
        for p in self.get_parameters().values():
            parameters+=p.size
        return parameters

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def forward_with_cache(self, *x)->(np.ndarray,Cache):
        pass

    def forward(self,*x):
        y,cache=self.forward_with_cache(*x)
        return y

    @abstractmethod
    def backward(self,δEδy:np.ndarray,cache:Cache)->(np.ndarray,ParameterSet):
        pass

    def __repr__(self):
        return f"{self.name}"

class ModelWithParameters(Model):

    def __init__(self,name=None):
        super().__init__(name=name)
        self.parameters = {}

    def register_parameter(self,name,value):
        self.parameters[name]=value

    def get_parameters(self):
        return self.parameters


class ErrorFunction:
    '''
    A
    '''
    def __init__(self,name=None):
        super().__init__(name=name)

    @abstractmethod
    def forward_with_cache(self, y:np.ndarray, y_pred:np.ndarray)->float:
        pass


    @abstractmethod
    def backward(self,cache:Cache):
        pass


class MeanError(ErrorFunction):
    '''
    Converts a Model that calculates
    an error function for each sample
    into a mean error function with a single scalar output
    which represents the final error of a network
    '''
    def __init__(self, sample_error:Model, name=None):
        super().__init__(name=name)
        self.sample_error_layer = sample_error

    def forward_with_cache(self, y_true:np.ndarray, y:np.ndarray):
        E, sample_cache=self.sample_error_layer.forward_with_cache(y_true, y)
        n=y_true.shape[0]
        cache =(n,sample_cache)
        return np.mean(E),cache

    def backward(self,cache):
        n,sample_cache=cache
        δEδy=np.ones(n)/n
        δEδy,δEδp=self.sample_error_layer.backward(δEδy,sample_cache)
        assert len(δEδy.shape)==1
        assert δEδy.shape[0] == n, "sample_error_layer's gradient must have n values"
        return δEδy
