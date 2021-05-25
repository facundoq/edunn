eps = 1e-12

from .model import ErrorModel,Model,Phase, ModelWithParameters
from .optimizer import StochasticGradientDescent
from . import initializers,plot

from .models import *
from . import metrics,datasets