import numpy as np
from edunn.model import Model, Phase


class Sequential(Model):
    """
        Models a neural network with a sequential (ie, linear) topology
        This network receives as input a single vector x
        And outputs a single vector y
    """

    def __init__(self, layers: [Model], name=None):
        """
        :param layers: List of models, in order
        """
        super().__init__(name)
        self.layers: [Model] = layers
        layer_names = [l.name for l in layers]
        layer_names_set = set(layer_names)
        assert len(layer_names) == len(layer_names_set), f"Layer names must be unique, found: {layer_names}"

    def forward(self, x: np.ndarray):
        """
        :param x: input to model
        :return: output of model with x as input
        """

        """ YOUR IMPLEMENTATION START """
        # default: y = np.zeros_like(x)
        for layer in self.layers:
            x = layer.forward(x)
        y = x
        """ YOUR IMPLEMENTATION END """
        return y

    def set_phase(self, phase: Phase):
        for l in self.layers:
            l.set_phase(phase)

    def merge_gradients(self, m_i, dE_dp_i, dE_dp):
        """
        :param m_i: model for ith layer
        :param dE_dp_i: derivatives parameters of model for ith layer
        :param dE_dp: all derivatives of parameters of Sequential
        """

        if not m_i.frozen:
            for k, v in dE_dp_i.items():
                new_name = self.generate_parameter_name(m_i, k)
                dE_dp[new_name] = v

    def backward(self, dE_dy: np.ndarray):
        """
        :param x: inputs
        :param y: expected output
        :return: gradients for every layer, prediction for inputs and error
        """
        dE_dx = 0
        dE_dp = {}
        # Hint: use `self.merge_gradients(m_i, dE_dp_i, dE_dp)`
        # to add the gradients `dE_dp_i` of parameters of `m_i`
        # to the final gradients `dE_dp` dictionary
        """ YOUR IMPLEMENTATION START """
        for m_i in reversed(self.layers):
            dE_dy, dE_dp_i = m_i.backward(dE_dy)
            self.merge_gradients(m_i, dE_dp_i, dE_dp)
        dE_dx = dE_dy
        """ YOUR IMPLEMENTATION END """

        return dE_dx, dE_dp

    def generate_parameter_name(self, l: Model, parameter_name: str):
        return f"{l.name}({parameter_name})"

    def get_parameters(self):
        parameters = {}
        for l in self.layers:
            for k, v in l.get_parameters().items():
                new_name = self.generate_parameter_name(l, k)
                parameters[new_name] = v
        return parameters

    def summary(self) -> str:
        """
        :return: a summary of the models of the model and their parameters
        """
        separator = "-------------------------------"
        result = f"{separator}\n"
        parameters = 0
        result += f"Model {self.name}:\n"
        for l in self.layers:
            layer_parameters = l.n_parameters()
            parameters += layer_parameters
            l_summary = f"{l.name} → params: {layer_parameters}"
            result += l_summary + "\n"
        result += f"Total parameters: {parameters}\n"
        result += f"{separator}\n"
        return result
