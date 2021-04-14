from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

class_counter = {}
from typing import Tuple, Dict


class Phase(Enum):
    Training = "Training"
    Test = "Test"


ParameterSet = Dict[str, np.ndarray]
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

    def __init__(self, name=None):
        if name is None:
            class_name = self.__class__.__name__
            count = class_counter.get(class_name, 0)
            name = f"{class_name}_{count}"
            class_counter[class_name] = count + 1
        self.phase = Phase.Training

        self.name = name
        self.frozen = False
        self.cache = None

    def set_cache(self, *values):
        self.cache = values

    def get_cache(self):
        return self.cache

    def set_phase(self, phase: Phase):
        '''
        Indicate to the model the current Phase (training/tests/other) the model is in.
        This can be used to code different behaviors in training and testing phases.
        :param phase: phase to set
        '''
        self.phase = phase

    def n_parameters(self) -> int:
        '''
        :return: number of individual parameters of the network
        '''
        result = 0
        for p in self.get_parameters().values():
            result += p.size
        return result

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def forward(self, *x) -> (np.ndarray):
        pass

    @abstractmethod
    def backward(self, δEδy: np.ndarray) -> (np.ndarray, ParameterSet):
        pass

    def __repr__(self):
        return f"{self.name}"


class ModelWithParameters(Model):
    '''
    Helper class to implement layers with parameters
    '''

    def __init__(self, name=None):
        super().__init__(name=name)
        self.parameters = {}

    def register_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameters(self):
        return self.parameters


class ModelWithoutParameters(Model):
    '''
    Helper class to implement layers _without_ parameters
    '''

    def __init__(self, name=None):
        super().__init__(name=name)

    def get_parameters(self):
        return {}


class ErrorModel(ModelWithoutParameters):
    '''
    Helper class to implement layers _without_ parameters
    '''

    def __init__(self, name=None):
        super().__init__(name=name)

    @abstractmethod
    def forward(self, y_true: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def backward(self, δE: float) -> (np.ndarray, ParameterSet):
        pass


class MeanError(ErrorModel):
    '''
    Converts a Model that converts
    an error function for each sample
    into a mean error function with a single scalar output
    which can represent the final error of a network
    '''

    def __init__(self, sample_error: Model, name=None):
        if name is None:
            name = f"Mean{sample_error.name}"
        super().__init__(name=name)
        self.sample_error_layer = sample_error

    def forward(self, y_true: np.ndarray, y: np.ndarray):
        E = self.sample_error_layer.forward(y_true, y)
        n = y_true.shape[0]
        self.set_cache(n)
        return np.mean(E)

    def backward(self, δE: float = 1) -> (np.ndarray, ParameterSet):
        '''

        :param δE:Scalar to scale the gradients
                Needed to comply with Model's interface
                Default is 1, which means no scaling
        :param cache: calculated from forward pass
        :return:
        '''
        n, = self.get_cache()
        # Since we just calculate the mean over n values
        # and the mean is equivalent to multiplying by 1/n
        # the gradient is simply 1/n for each value
        δEδy = np.ones((n, 1)) / n
        # Return error gradient scaled by δE
        δEδy *= δE
        # Calculate gradient for each sample
        δEδy_sample, δEδp_sample = self.sample_error_layer.backward(δEδy)
        assert_message = f"sample_error_layer's gradient must have n values (found {δEδy_sample.shape[0]}, expected {n})"
        assert δEδy_sample.shape[0] == n, assert_message
        return δEδy_sample, δEδp_sample
