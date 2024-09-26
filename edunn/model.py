from abc import ABC, abstractmethod
import numpy as np
from enum import Enum
from typing import Tuple, Dict


class Phase(Enum):
    Training = "Training"
    Test = "Test"


ParameterSet = Dict[str, np.ndarray]
Cache = Tuple


class Model(ABC):
    """
    A Model can perform forward and backward operations
    Each model has a unique name. Also, it has a set of parameters that can be updated.
    The forward operation takes an input (x) and generates an output (y)
    The backward takes an error gradient for the output (dE_dy) and generates error gradients for the input (dE_dx) and parameters (dE_dp)
    The backward operation must be defined explicitly and is not automatically derived from the forward.
    The model can be frozen, so that its parameters are not updated/optimized.
    """

    model_name_registry = []
    class_counter = {}

    def __init__(self, name=None):
        """
        Create a model with a name.
        Names should be unique so that gradients from different models can be updated independently.
        """
        if name is None:
            # auto generate a name if not provided
            class_name = self.__class__.__name__
            count = self.class_counter.get(class_name, 0)
            name = f"{class_name}_{count}"
            self.class_counter[class_name] = count + 1
        assert not (name in self.model_name_registry), \
            f"The model name {name} has already been used, see model_name_registry: {self.model_name_registry}."
        self.model_name_registry.append(name)

        self.phase = Phase.Training

        self.name = name
        self.frozen = False
        self.cache = None

        self.parameters = {}

    def set_cache(self, *values):
        self.cache = values

    def get_cache(self):
        return self.cache

    def set_phase(self, phase: Phase):
        """
        Indicate to the model the current Phase (training/tests/other) the model is in.
        This can be used to code different behaviors in training and testing phases.
        :param phase: phase to set
        """
        self.phase = phase

    def n_parameters(self) -> int:
        """
        :return: number of individual parameters of the network
        """
        result = 0
        for p in self.get_parameters().values():
            result += p.size
        return result

    def register_parameter(self, name, value):
        self.parameters[name] = value

    def get_parameters(self):
        return self.parameters

    @abstractmethod
    def forward(self, *x) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dE_dy: np.ndarray) -> (np.ndarray, ParameterSet):
        pass

    def __repr__(self):
        return f"{self.name}"
