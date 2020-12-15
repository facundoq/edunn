from abc import ABC,abstractmethod
import numpy as np
from enum import Enum
layer_counter = {}
from typing import Tuple,Dict

class Phase(Enum):
    Training = "Training"
    Test = "Test"
ParameterSet = Dict[str,np.ndarray]
Cache = Tuple
class Layer(ABC):
    '''
    A layer of a Model. Can perform forward and backward operations
    The backward operation must be defined explicitely and is not automatically derived from the forward.
    Each layer has a unique name. Also, it has a set of parameters that can be updated to fit a model.
    The layer can be frozen, so that its parameters are not updated/optimized.
    Finally, the layer can be stateful, so the reset() method can reinitialize the layer.
    '''
    def __init__(self,name=None):
        if name is None:
            class_name= self.__class__.__name__
            count=layer_counter.get(class_name,0)
            name =f"{class_name}_{count}"
            layer_counter[class_name]=count+1
        self.phase=Phase.Training

        self.name=name
        self.parameters = {}
        self.frozen=False


    def set_phase(self,phase:Phase):
        '''
        Indicate to the layer the current Phase (training/test/other) the model is in.
        This can be used to code different behaviors in training and testing phases.
        :param phase: phase to set
        '''
        self.phase = phase

    def n_parameters(self)->int:
        parameters=0
        for p in self.get_parameters().values():
            parameters+=p.size
        return parameters

    def register_parameter(self,name,value):
        self.parameters[name]=value

    def get_parameters(self):
        return self.parameters

    @abstractmethod
    def forward_with_cache(self, *x):
        pass

    def forward(self,*x):
        y,cache=self.forward_with_cache(*x)
        return y

    @abstractmethod
    def backward(self,*x):
        pass

    def __repr__(self):
        return f"{self.name}"

class CommonLayer(Layer):

    def __init__(self,name=None):
        super().__init__(name=name)

    @abstractmethod
    def forward_with_cache(self, x:np.ndarray)->(np.ndarray,Cache):
        pass


    @abstractmethod
    def backward(self,δEδy:np.ndarray,cache:Cache)->(np.ndarray,ParameterSet):
        pass

class ErrorLayer(Layer):

    def __init__(self,name=None):
        super().__init__(name=name)

    @abstractmethod
    def forward_with_cache(self, y:np.ndarray, y_pred:np.ndarray)->float:
        pass


    @abstractmethod
    def backward(self,cache:Cache):
        pass



class SampleErrorLayer(Layer):

    def __init__(self,name=None):
        super().__init__(name=name)

    @abstractmethod
    def forward_with_cache(self, y:np.ndarray, y_pred:np.ndarray)->np.ndarray:
        pass


    @abstractmethod
    def backward(self,cache:Cache):
        pass


class MeanError(ErrorLayer):
    '''
    Converts a SampleErrorLayer that calculates
    an error function for each sample
    into a mean error function with a single scalar output
    which represents the final error of a network
    '''
    def __init__(self, e:SampleErrorLayer,name=None):
        super().__init__(name=name)
        self.e = e

    def forward_with_cache(self, y_true:np.ndarray, y:np.ndarray):
        E, sample_cache=self.e.forward_with_cache(y_true, y)
        n,dout=y_true.shape
        cache =(n,sample_cache)
        return np.mean(E),cache

    def backward(self,cache):
        n,sample_cache=cache
        δEδy,δEδp=self.e.backward(sample_cache)
        assert δEδy.shape[0] == n, "SampleErrorLayer's gradient must have n values"
        return δEδy/n
