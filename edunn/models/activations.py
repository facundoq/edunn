"""
Typically, Activations are layers without parameters, applied element-wise.
"""

from edunn.model import Model
import numpy as np
import math


class Identity(Model):

    def forward(self, x: np.ndarray):
        return x

    def backward(self, dE_dy: np.ndarray):
        # no parameters, no derivatives
        dE_dp = {}
        return dE_dy, dE_dp


class AddConstant(Model):
    """
    A layer that adds a constant
    This layer has NO parameters
    """

    def __init__(self, value: float, name=None):
        super().__init__(name=name)
        self.value = value

    def forward(self, x: np.ndarray):
        """
        :param x: input vector/matrix
        :return: `x + a`, constant value, stored in `self.value`
        """

        """ YOUR IMPLEMENTATION START """
        # default: y = np.zeros_like(x)
        y = x + self.value
        """ YOUR IMPLEMENTATION END """

        return y

    def backward(self, dE_dy: np.ndarray):
        """ YOUR IMPLEMENTATION START """
        # default: dE_dx = np.zeros_like(dE_dy)
        dE_dx = dE_dy
        """ YOUR IMPLEMENTATION END """

        dE_dp = {}  # no parameters, no derivatives
        return dE_dx, dE_dp


class ReLU(Model):

    def forward(self, x: np.ndarray):
        y = np.zeros_like(x)

        # TIP: NO utilizar np.max()
        # Ya que devuelve el valor máximo y no aplica la función elemento a elemento

        """ YOUR IMPLEMENTATION START """
        y = np.maximum(x, 0)
        """ YOUR IMPLEMENTATION END """
        self.set_cache(y)
        return y

    def backward(self, dE_dy: np.ndarray):
        dE_dx = np.zeros_like(dE_dy)
        y, = self.get_cache()

        # TIP: dE_dx = dE_dy * dy_dx
        # dy_dx is 1 if the output was greater than 0, and 0 otherwise

        """ YOUR IMPLEMENTATION START """
        dy_dx = y > 0
        dE_dx = dE_dy * dy_dx
        """ YOUR IMPLEMENTATION END """

        return dE_dx, {}


@np.vectorize
def erf(x):
    return math.erf(x)


def normal_pdf(x, mean=0, std=1):
    # Calculate the probability density function of the normal distribution
    pdf = np.zeros_like(x)
    """ YOUR IMPLEMENTATION START """
    pdf = (1 / (np.sqrt(2 * np.pi * std ** 2))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    """ YOUR IMPLEMENTATION END """
    return pdf


class GELU(Model):

    def forward(self, x: np.ndarray):
        y = np.zeros_like(x)
        cache = tuple()
        """ YOUR IMPLEMENTATION START """
        cdf = 0.5 * (1 + erf(x / 2.0 ** 0.5))
        y = x * cdf
        cache = (x, cdf)
        """ YOUR IMPLEMENTATION END """
        self.set_cache(cache)
        return y

    def backward(self, dE_dy: np.ndarray):
        dE_dx = np.zeros_like(dE_dy)
        (x, cdf), = self.get_cache()
        """ YOUR IMPLEMENTATION START """
        pdf_val = normal_pdf(x, 0, 1)
        dE_dx = dE_dy * (cdf + x * pdf_val)
        """ YOUR IMPLEMENTATION END """
        return dE_dx, {}


class Sigmoid(Model):

    def forward(self, x: np.ndarray):
        y = np.zeros_like(x)
        """ YOUR IMPLEMENTATION START """
        y = 1.0 / (1.0 + np.exp(-x))
        """ YOUR IMPLEMENTATION END """
        cache = (y,)
        self.set_cache(y)
        return y

    def backward(self, dE_dy: np.ndarray):
        dE_dx = np.zeros_like(dE_dy)
        y, = self.get_cache()
        # TIP: dE_dx = dE_dy * dy_dx
        # First calculate dy_dx
        # then multiply by dE_dy (provided)

        """ YOUR IMPLEMENTATION START """
        dy_dx = y * (1.0 - y)
        dE_dx = dE_dy * dy_dx
        """ YOUR IMPLEMENTATION END """

        return dE_dx, {}


class TanH(Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.sigmoid = Sigmoid()

    def forward(self, x: np.ndarray):
        y = np.zeros_like(x)
        # TIP: TanH2 is simply sigmoid*2-1
        # we already defined self.sigmoid as a sigmod Layer
        # use it to simplify your implementation
        """ YOUR IMPLEMENTATION START """
        s = self.sigmoid.forward(2 * x)
        y = s * 2 - 1
        """ YOUR IMPLEMENTATION END """
        return y  # this layer's cache is the same as the sigmoid's cache

    def backward(self, dE_dy: np.ndarray):
        dE_dx = np.zeros_like(dE_dy)
        # TIP: If TanH2 is simply sigmoid*2-1
        # Calculate derivative of TanH
        # in terms of derivative of sigmoid
        """ YOUR IMPLEMENTATION START """
        dE_dx, dE_dp = self.sigmoid.backward(dE_dy)
        dE_dx = dE_dx * 4
        """ YOUR IMPLEMENTATION END """

        return dE_dx, {}


class Softmax(Model):
    def __init__(self, name=None, smoothing=1e-16):
        super().__init__(name)
        self.smoothing = smoothing

    def forward(self, x: np.ndarray):
        # add a small value so that no probability ends up exactly 0
        # This avoids NaNs when computing log(p) or 1/p
        # Specially when paired with the CrossEntropy error function
        x = x + self.smoothing

        n, classes = x.shape
        y = np.zeros_like(x)
        for i in range(n):
            xi = x[i, :]
            xi = xi + xi.max()  # trick to avoid numerical issues
            # Calcular las probabilidades para cada clase
            # y guardar el valor en y[i,:] en base al vector de puntaje xi
            # Nota: este cálculo es para 1 ejemplo del batch el for se encarga de repetirlo para c/u
            """ YOUR IMPLEMENTATION START """
            e = np.exp(xi)
            N = e.sum()
            y[i, :] = e / N
            """ YOUR IMPLEMENTATION END """
        self.set_cache(y)
        return y

    def backward(self, dE_dY: np.ndarray):
        # dE_dx = dE_dY * dy_dx
        y, = self.get_cache()
        n, classes = dE_dY.shape
        dE_dx = np.zeros_like(dE_dY)
        for i in range(n):
            dE_dx[i, :] = self.backward_sample(dE_dY[i, :], y[i, :])
        return dE_dx, {}

    def backward_sample(self, dE_dy: np.ndarray, y: np.ndarray):
        # AYUDA PARA EL CÁLCULO
        # http://facundoq.github.io/guides/softmax_derivada.html
        """
        :param dE_dy: derivative of error wrt output for a *single sample*
        :param y: output for a *single sample*
        :return: dE_dx for a *single sample*
        """
        classes = y.shape[0]

        # AYUDA PARA EL CÁLCULO
        # http://facundoq.github.io/guides/softmax_derivada.html
        dy_dx = np.zeros((classes, classes))
        """ YOUR IMPLEMENTATION START """
        for i in range(classes):
            for j in range(classes):
                if i == j:
                    dy_dx[i, j] = (1 - y[i]) * y[j]
                else:
                    dy_dx[i, j] = -y[i] * y[j]

        # # Vectorized Version
        # id = np.identity(classes)
        # y = y[:,np.newaxis]
        # A = y.repeat(classes,axis=1)
        # B = y.T.repeat(classes,axis=0)
        # dy_dx = (id-A)*B

        """ YOUR IMPLEMENTATION END """

        dE_dx = np.zeros_like(dE_dy)
        classes = y.shape[0]
        for j in range(classes):
            dy_dx_j = dy_dx[:, j]

            """ YOUR IMPLEMENTATION START """
            dE_dx[j] = dE_dy.dot(dy_dx_j)
            """ YOUR IMPLEMENTATION END """
        return dE_dx
