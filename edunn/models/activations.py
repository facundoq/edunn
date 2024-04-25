from edunn.model import ModelWithParameters,Cache,ModelWithoutParameters
import numpy as np
from scipy.special import erf

class Identity(ModelWithParameters):

    def forward(self, x:np.ndarray):
        return x

    def backward(self,δEδy:np.ndarray):
        δEδp={} # no parameters, no derivatives
        return δEδy,δEδp

class AddConstant(ModelWithParameters):
    '''
    A layer that adds a constant
    This layer has NO parameters
    '''

    def __init__(self,value:float,name=None):
        super().__init__(name=name)
        self.value=value

    def forward(self, x:np.ndarray):
        '''
        :param x: input vector/matrix
        :return: x + a constant value, stored in self.value
        '''
        y= np.zeros_like(x)
        ### YOUR IMPLEMENTATION START  ###

        y=x+self.value
        ### YOUR IMPLEMENTATION END  ###
        return y

    def backward(self,δEδy:np.ndarray):
        δEδx= np.zeros_like(δEδy)
        ### YOUR IMPLEMENTATION START  ###
        δEδx= δEδy
        ### YOUR IMPLEMENTATION END  ###
        δEδp={} # no parameters, no derivatives
        return δEδx,δEδp

class MultiplyConstant(ModelWithParameters):
    '''
    A layer that multiplies by a constant
    This layer has NO parameters
    '''
    def __init__(self,value:float,name=None):
        super().__init__(name=name)
        self.value=value
    def forward(self, x:np.ndarray):
        '''
        :param x: input vector/matrix
        :return: x * a constant value, stored in self.value
        '''
        y= np.zeros_like(x)
        ### YOUR IMPLEMENTATION START  ###
        y=x*self.value
        ### YOUR IMPLEMENTATION END  ###
        return y


    def backward(self,δEδy:np.ndarray):
        δEδx= np.zeros_like(δEδy)

        ### YOUR IMPLEMENTATION START  ###
        δEδx=δEδy *self.value
        ### YOUR IMPLEMENTATION END  ###

        δEδp={} # no parameters, no derivatives
        return δEδx,δEδp



class ReLU(ModelWithoutParameters):

    def forward(self, x:np.ndarray):
        y = np.zeros_like(x)

        # TIP: NO utilizar np.max()
        # Ya que devuelve el valor máximo
        # y no aplica la función elemento a elemento

        ### YOUR IMPLEMENTATION START  ###
        y = np.maximum(x,0)
        ### YOUR IMPLEMENTATION END  ###
        self.set_cache(y)
        return y

    def backward(self, δEδy:np.ndarray):
        δEδx = np.zeros_like(δEδy)
        y, = self.get_cache()

        # TIP: δEδx = δEδy * δyδx
        # δyδx is 1 if the output was greater than 0
        # and 0 otherwise

        ### YOUR IMPLEMENTATION START  ###
        δyδx = y>0
        δEδx = δEδy * δyδx
        ### YOUR IMPLEMENTATION END  ###

        return δEδx,{}


def normal_pdf(x, mean=0, std=1):
    # Calculate the probability density function of the normal distribution
    pdf = np.zeros_like(x)
    ### YOUR IMPLEMENTATION START  ###
    pdf = (1 / (np.sqrt(2 * np.pi * std**2))) * np.exp(-((x - mean)**2) / (2 * std**2))
    ### YOUR IMPLEMENTATION END  ###
    return pdf

class GELU(ModelWithoutParameters):

    def forward(self, x:np.ndarray):
        y = np.zeros_like(x)
        cache = tuple()
        ### YOUR IMPLEMENTATION START  ###
        cdf = 0.5 * (1 + erf(x / 2.0**0.5))
        y = x * cdf
        cache = (x, cdf)
        ### YOUR IMPLEMENTATION END  ###
        self.set_cache(cache)
        return y

    def backward(self, δEδy:np.ndarray):
        δEδx = np.zeros_like(δEδy)
        (x, cdf), = self.get_cache()
        ### YOUR IMPLEMENTATION START  ###
        pdf_val = normal_pdf(x, 0, 1)
        δEδx = δEδy * (cdf + x * pdf_val)
        ### YOUR IMPLEMENTATION END  ###
        return δEδx,{}

class Sigmoid(ModelWithoutParameters):

    def forward(self, x:np.ndarray):
        y = np.zeros_like(x)
        ### YOUR IMPLEMENTATION START  ###
        y =   1.0/(1.0 + np.exp(-x))
        ### YOUR IMPLEMENTATION END  ###
        cache = (y,)
        self.set_cache(y)
        return y

    def backward(self, δEδy:np.ndarray):
        δEδx= np.zeros_like(δEδy)
        y, = self.get_cache()
        # TIP: δEδx = δEδy * δyδx
        # First calculate δyδx
        # then multiply by δEδy (provided)

        ### YOUR IMPLEMENTATION START  ###
        δyδx = y * (1.0-y)
        δEδx = δEδy * δyδx
        ### YOUR IMPLEMENTATION END  ###

        return δEδx,{}



class TanH(ModelWithoutParameters):
    def __init__(self,name=None):
        super().__init__(name=name)
        self.sigmoid=Sigmoid()

    def forward(self, x:np.ndarray):
        y= np.zeros_like(x)
        # TIP: TanH2 is simply sigmoid*2-1
        # we already defined self.sigmoid as a sigmod Layer
        # use it to simplify your implementation
        ### YOUR IMPLEMENTATION START  ###
        s = self.sigmoid.forward(2*x)
        y= s * 2 - 1
        ### YOUR IMPLEMENTATION END  ###
        return y # this layer's cache is the same as the sigmoid's cache

    def backward(self,δEδy:np.ndarray):
        δEδx= np.zeros_like(δEδy)
        # TIP: If TanH2 is simply sigmoid*2-1
        # Calculate derivative of TanH
        # in terms of derivative of sigmoid
        ### YOUR IMPLEMENTATION START  ###
        δEδx,δEδp =self.sigmoid.backward(δEδy)
        δEδx = δEδx*4
        ### YOUR IMPLEMENTATION END  ###

        return δEδx,{}


class Softmax(ModelWithoutParameters):
    def __init__(self,name=None,smoothing=1e-16):
        super().__init__(name)
        self.smoothing=smoothing

    def forward(self, x:np.ndarray):
        # add a small value so that no probability ends up exactly 0
        # This avoids NaNs when computing log(p) or 1/p
        # Specially when paired with the CrossEntropy error function
        x=x+self.smoothing

        n,classes=x.shape
        y = np.zeros_like(x)
        for i in range(n):
            xi=x[i,:]
            xi = xi + xi.max() # trick to avoid numerical issues
            # Calcular las probabilidades para cada clase
            # y guardar el valor en y[i,:]
            # en base al vector de puntaje xi
            # Nota: este cálculo es para 1 ejemplo del batch
            # el for se encarga de repetirlo para c/u
            ### YOUR IMPLEMENTATION START  ###
            e = np.exp(xi)
            N = e.sum()
            y[i,:] = e/N
            ### YOUR IMPLEMENTATION END  ###
        self.set_cache(y)
        return y

    def backward(self, δEδy:np.ndarray):
        # δEδx = δEδy * δyδx
        y, = self.get_cache()
        n,classes = δEδy.shape
        δEδx = np.zeros_like(δEδy)
        for i in range(n):
            δEδx[i,:]= self.backward_sample(δEδy[i,:],y[i,:])
        return δEδx,{}


    def backward_sample(self,δEδy:np.ndarray,y:np.ndarray):
        # AYUDA PARA EL CÁLCULO
        # http://facundoq.github.io/guides/softmax_derivada.html
        '''
        :param δEδy: derivative of error wrt output for a *single sample*
        :param y: output for a *single sample*
        :return: δEδx for a *single sample*
        '''
        classes=y.shape[0]

        # AYUDA PARA EL CÁLCULO
        # http://facundoq.github.io/guides/softmax_derivada.html
        δyδx = np.zeros((classes,classes))
        ### YOUR IMPLEMENTATION START  ###
        for i in range(classes):
            for j in range(classes):
                if i == j:
                    δyδx[i,j]=(1-y[i])*y[j]
                else:
                    δyδx[i,j]=-y[i]*y[j]

        # # Vectorized Version
        # id = np.identity(classes)
        # y = y[:,np.newaxis]
        # A = y.repeat(classes,axis=1)
        # B = y.T.repeat(classes,axis=0)
        # δyδx = (id-A)*B

        ### YOUR IMPLEMENTATION END  ###

        δEδx = np.zeros_like(δEδy)
        classes=y.shape[0]
        for j in range(classes):
            δyδx_j = δyδx[:,j]

            ### YOUR IMPLEMENTATION START  ###
            δEδx[j] = δEδy.dot(δyδx_j)
            ### YOUR IMPLEMENTATION END  ###
        return δEδx

