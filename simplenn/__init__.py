eps = 1e-12

from .model import ErrorModel,Model,Phase,MeanError,ModelWithParameters
from .optimizer import StochasticGradientDescent
from . import initializers,metrics,plot

from .models import *