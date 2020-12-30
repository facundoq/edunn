from simplenn.model import ModelWithParameters,Cache
import numpy as np

class Identity(ModelWithParameters):

    def forward_with_cache(self, x:np.ndarray):
        cache = tuple() # empty tuple
        return x,cache

    def backward(self,δEδy:np.ndarray,cache:Cache):
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

    def forward_with_cache(self, x:np.ndarray):
        '''
        :param x: input vector/matrix
        :return: x + a constant value, stored in self.value
        '''
        y= np.zeros_like(x)
        ### COMPLETAR INICIO ###

        y=x+self.value
        ### COMPLETAR FIN ###
        cache = tuple() # empty tuple
        return y,cache

    def backward(self,δEδy:np.ndarray,cache:Cache):
        δEδx= np.zeros_like(δEδy)
        ### COMPLETAR INICIO ###
        δEδx= δEδy
        ### COMPLETAR FIN ###
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
    def forward_with_cache(self, x:np.ndarray):
        '''
        :param x: input vector/matrix
        :return: x * a constant value, stored in self.value
        '''
        y= np.zeros_like(x)
        ### COMPLETAR INICIO ###
        y=x*self.value
        ### COMPLETAR FIN ###
        cache = tuple() # empty tuple
        return y,cache


    def backward(self,δEδy:np.ndarray,cache:Cache):
        δEδx= np.zeros_like(δEδy)

        ### COMPLETAR INICIO ###
        δEδx=δEδy *self.value
        ### COMPLETAR FIN ###

        δEδp={} # no parameters, no derivatives
        return δEδx,δEδp



class ReLU(ModelWithParameters):

    def forward_with_cache(self, x:np.ndarray):
        y = np.zeros_like(x)

        # TIP: NO utilizar np.max()
        # Ya que devuelve el valor máximo
        # y no aplica la función elemento a elemento

        ### COMPLETAR INICIO ###
        y = np.maximum(x,0)
        ### COMPLETAR FIN ###
        cache = (y,)
        return y,cache

    def backward(self, δEδy:np.ndarray,cache:Cache):
        δEδx = np.zeros_like(δEδy)
        y, = cache

        # TIP: δEδx = δEδy * δyδx
        # δyδx is 1 if the output was greater than 0
        # and 0 otherwise

        ### COMPLETAR INICIO ###
        δyδx = y>0
        δEδx = δEδy * δyδx
        ### COMPLETAR FIN ###

        return δEδx,{}


class Sigmoid(ModelWithParameters):

    def forward_with_cache(self, x:np.ndarray):
        y = np.zeros_like(x)
        ### COMPLETAR INICIO ###
        y =   1.0/(1.0 + np.exp(-x))
        ### COMPLETAR FIN ###
        cache = (y,)
        return y,cache

    def backward(self, δEδy:np.ndarray,cache:Cache):
        δEδx= np.zeros_like(δEδy)
        y, = cache
        # TIP: δEδx = δEδy * δyδx
        # First calculate δyδx
        # then multiply by δEδy (provided)

        ### COMPLETAR INICIO ###
        δyδx = y * (1.0-y)
        δEδx = δEδy * δyδx
        ### COMPLETAR FIN ###

        return δEδx,{}



class TanH(ModelWithParameters):
    def __init__(self,name=None):
        self.sigmoid=Sigmoid()
        super().__init__(name=name)


    def forward_with_cache(self, x:np.ndarray):
        y= np.zeros_like(x)
        # TIP: TanH is simply sigmoid*2-1
        ### COMPLETAR INICIO ###
        s ,cache = self.sigmoid.forward_with_cache(x)
        y= s * 2 - 1
        ### COMPLETAR FIN ###
        return y,cache # this layer's cache is the same as the sigmoid's cache

    def backward(self,δEδy:np.ndarray,cache:Cache):
        δEδx= np.zeros_like(δEδy)
        # TIP: If TanH is simply sigmoid*2-1
        # Calculate derivative of TanH
        # in terms of derivative of sigmoid
        ### COMPLETAR INICIO ###
        δEδx,δEδp =self.sigmoid.backward(δEδy,cache)
        δEδx = δEδx*2
        ### COMPLETAR FIN ###

        return δEδx,{}


class Softmax(ModelWithParameters):
    def __init__(self,name=None,smoothing=1e-12):
        super().__init__(name)
        self.smoothing=smoothing

    def forward_with_cache(self, x:np.ndarray):
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
            ### COMPLETAR INICIO ###
            e = np.exp(xi)
            y[i,:] = e/e.sum()
            ### COMPLETAR FIN ###
        cache = (y,)
        return y,cache

    def backward(self, δEδy:np.ndarray,cache:Cache):
        # δEδx = δEδy * δyδx
        y, = cache
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
        ### COMPLETAR INICIO ###
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

        ### COMPLETAR FIN ###

        δEδx = np.zeros_like(δEδy)
        classes=y.shape[0]
        for j in range(classes):
            δyδx_j = δyδx[:,j]

            ### COMPLETAR INICIO ###
            δEδx[j] = δEδy.dot(δyδx_j)
            ### COMPLETAR FIN ###
        return δEδx

