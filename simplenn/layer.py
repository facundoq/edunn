from abc import ABC,abstractmethod
import numpy as np
from enum import Enum
layer_counter = {}

class Phase(Enum):
    Training = "Training"
    Test = "Test"

class Layer(ABC):

    def __init__(self,name=None):
        if name is None:
            class_name= self.__class__.__name__
            count=layer_counter.get(class_name,0)
            name =f"{class_name}_{count}"
            layer_counter[class_name]=count+1
        self.phase=Phase.Training

        self.name=name
        self.parameters = {}
        self.reset()
        self.frozen=False

    def set_phase(self,phase:Phase):
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

    def reset(self):
        self.cache = ()

    def set_cache(self,*args):
        self.cache = args

    @abstractmethod
    def forward(self,x:np.ndarray):
        pass


    @abstractmethod
    def backward(self,δEδy:np.ndarray):
        pass

    def __repr__(self):
        return f"{self.name}"

class ErrorLayer(ABC):


    @abstractmethod
    def forward(self,y:np.ndarray,y_pred:np.ndarray)->float:
        pass


    @abstractmethod
    def backward(self,y:np.ndarray,y_pred:np.ndarray):
        pass


class SampleErrorLayer(ABC):


    @abstractmethod
    def forward(self,y:np.ndarray,y_pred:np.ndarray)->np.ndarray:
        pass


    @abstractmethod
    def backward(self,y:np.ndarray,y_pred:np.ndarray):
        pass


class MeanError(ErrorLayer):
    '''
    Converts a SampleErrorLayer that calculates
    an error function for each sample
    into a mean error function with a single scalar output
    which represents the final error of a network
    '''
    def __init__(self, e:SampleErrorLayer):
        super().__init__()
        self.e = e

    def forward(self,y:np.ndarray,y_pred:np.ndarray):
        return np.mean(self.e.forward(y,y_pred))

    def backward(self,y:np.ndarray,y_pred:np.ndarray):
        n,dout=y.shape
        δEδy=self.e.backward(y,y_pred)/n
        assert δEδy.shape[0] == n, "SampleErrorLayer's gradient must have n values"

        return δEδy