import abc
import numpy as np
from typing import Tuple


class Initializer(abc.ABC):

    @abc.abstractmethod
    def initialize(self, p: np.ndarray):
        pass

    def create(self, shape: Tuple[int, ...]) -> np.ndarray:
        p = np.zeros(shape)
        self.initialize(p)
        return p


class Zero(Initializer):
    def initialize(self, p: np.ndarray):
        p[:] = 0


class Constant(Initializer):
    def __init__(self, c: np.ndarray):
        self.c = c

    def initialize(self, p: np.ndarray):
        """ YOUR IMPLEMENTATION START """
        p[:] = self.c
        """ YOUR IMPLEMENTATION END """


class RandomUniform(Initializer):
    """
    A random initializer with samples from a Uniform distribution
    with mean 0 and range -a,a
    """

    def __init__(self, a: float = 1e-10):
        super().__init__()
        self.a = a

    def initialize(self, p: np.ndarray):
        # en: TIP use np.random.uniform(a,b,shape) to generate uniform random numbers between a and b
        # es: TIP usar np.random.uniform(a,b,shape) para generar numeros aleatorios uniformes entre a y b

        """ YOUR IMPLEMENTATION START """
        p[:] = np.random.uniform(-self.a, self.a, p.shape)
        """ YOUR IMPLEMENTATION END """


class RandomNormal(Initializer):
    def __init__(self, std: float = 1e-10):
        super().__init__()
        self.std = std

    def initialize(self, p: np.ndarray):
        # en: TIP use np.random.normal(μ, σ, shape) to generate random numbers
        # en: sampled from a normal distribution with mean μ and std σ
        # es: TIP usar np.random.normal(μ, σ, shape) para generar numeros aleatorios
        # es: muestreados de una distribucion normal con media μ y desvio σ

        """ YOUR IMPLEMENTATION START """
        p[:] = np.random.normal(0, self.std, p.shape)
        """ YOUR IMPLEMENTATION END """


class XavierUniform(Initializer):

    def initialize(self, p: np.ndarray):
        din, dout = p.shape
        # en: TIP use np.random.random_sample(shape) to generate uniform randomnumbers between 0 and 1
        # es: TIP usar np.random.random_sample(shape) para generar numeros aleatorios entre 0 y 1
        """ YOUR IMPLEMENTATION START """
        factor = np.sqrt(6 / (din + dout))
        p[:] = (np.random.random_sample(p.shape) * 2 - 1) * factor
        """ YOUR IMPLEMENTATION END """


class XavierNormal(Initializer):

    def initialize(self, p: np.ndarray):
        din, dout = p.shape
        # en: TIP use np.random.normal(mu,std,shape) to sample random numbers from a normal distribution
        # es: TIP usar np.random.normal(mu,std,shape) para muestrear numeros aleatorios de una distribucion normal

        """ YOUR IMPLEMENTATION START """
        std = np.sqrt(6 / (din + dout))
        p[:] = (np.random.normal(0, std, p.shape))
        """ YOUR IMPLEMENTATION END """


class KaimingNormal(Initializer):

    def initialize(self, p: np.ndarray):
        din, *dout = p.shape
        # en: TIP use np.random.normal(mu,std,shape) to sample random numbers from a normal distribution
        # es: TIP usar np.random.normal(mu,std,shape) para muestrear numeros aleatorios de una distribucion normal

        """ YOUR IMPLEMENTATION START """
        std = np.sqrt(2 / din)
        p[:] = (np.random.normal(0, std, p.shape))
        """ YOUR IMPLEMENTATION END """


class KaimingUniform(Initializer):

    def initialize(self, p: np.ndarray):
        din, *dout = p.shape
        # en: TIP use np.random.random_sample(shape) to generate uniform random numbers between 0 and 1
        # es: TIP usar np.random.random_sample(shape) para generar numeros aleatorios entre 0 y 1

        """ YOUR IMPLEMENTATION START """
        factor = np.sqrt(2 / din)
        p[:] = (np.random.random_sample(p.shape) * 2 - 1) * factor
        """ YOUR IMPLEMENTATION END """
