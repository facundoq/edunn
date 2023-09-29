import numpy as np

from edunn import ModelWithoutParameters, Model
from edunn.model import ParameterSet


class MeanError(ModelWithoutParameters):
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
        E = 0
        ### YOUR IMPLEMENTATION START  ###
        Ei = self.sample_error_layer.forward(y_true, y)
        E = np.mean(Ei)
        ### YOUR IMPLEMENTATION END  ###
        n = y_true.shape[0]
        self.set_cache(n)
        return E

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