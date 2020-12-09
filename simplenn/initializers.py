import abc
import numpy as np
from typing import Tuple


class Initializer(abc.ABC):

    @abc.abstractmethod
    def initialize(self,p:np.ndarray):
        pass

    def create(self,shape:Tuple[int,...]):
        p = np.zeros(shape)
        self.initialize(p)
        return p

class Zero(Initializer):
    def initialize(self,p:np.ndarray):
        p[:]=0


class Constant(Initializer):
    def __init__(self,c:np.ndarray):
        self.c=c

    def initialize(self,p:np.ndarray):
        ### COMPLETAR INICIO ###
        p[:]=self.c
        ### COMPLETAR FIN ###


class RandomUniform(Initializer):
    '''
    A random initializer with samples from a Uniform distribution
    with mean 0 and range -a,a
    '''
    def __init__(self, a:float=1e-10):
        super().__init__()
        self.a=a

    def initialize(self,p:np.ndarray):
        #TIP use np.random.uniform(a,b,shape)
        #to generate uniform random numbers between a and b

        ### COMPLETAR INICIO ###
        p[:]=np.random.uniform(-self.a,self.a,p.shape)
        ### COMPLETAR FIN ###


class RandomNormal(Initializer):
    def __init__(self, std:float=1e-10):
        super().__init__()
        self.std=std

    def initialize(self,p:np.ndarray):
        #TIP use np.random.normal(μ,σ,shape)
        # to generate random numbers sampled from
        # a normal distribution with mean μ and std σ

        ### COMPLETAR INICIO ###
        p[:]=np.random.normal(0,self.std,p.shape)
        ### COMPLETAR FIN ###


class XavierUniform(Initializer):

    def initialize(self,p:np.ndarray):
        din,dout=p.shape
        #TIP use np.random.random_sample(shape)
        # to generate uniform randomnumbers between 0 and 1
        ### COMPLETAR INICIO ###
        factor = np.sqrt(6/(din+dout))
        p[:]=(np.random.random_sample(p.shape)*2-1)*factor
        ### COMPLETAR FIN ###

class XavierNormal(Initializer):

    def initialize(self,p:np.ndarray):
        din,dout=p.shape
        #TIP use np.random.normal(mu,std,shape)
        # to sample random numbers from a normal distribution

        ### COMPLETAR INICIO ###
        std = np.sqrt(6/(din+dout))
        p[:]=(np.random.normal(0,std, p.shape))
        ### COMPLETAR FIN ###

class KaimingNormal(Initializer):

    def initialize(self,p:np.ndarray):
        din,dout=p.shape
        #TIP use np.random.normal(mu,std,shape)
        # to sample random numbers from a normal distribution

        ### COMPLETAR INICIO ###
        std = np.sqrt(2/din)
        p[:]=(np.random.normal(0,std, p.shape))
        ### COMPLETAR FIN ###

class KaimingUniform(Initializer):

    def initialize(self,p:np.ndarray):
        din,dout=p.shape
        #TIP use np.random.random_sample(shape)
        # to generate uniform randomnumbers between 0 and 1

        ### COMPLETAR INICIO ###
        factor = np.sqrt(2/din)
        p[:]=(np.random.random_sample(p.shape)*2-1)*factor
        ### COMPLETAR FIN ###