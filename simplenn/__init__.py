eps = 1e-12

from .model import ErrorModel,Model,Phase,MeanError,ModelWithParameters
from .models import Linear,Bias,Sigmoid,TanH,ReLU,MultiplyConstant,AddConstant,Softmax,SquaredError,CrossEntropyWithLabels,BinaryCrossEntropyWithLabels,Dense,Sequential

from .optimizer import StochasticGradientDescent
from . import initializers,metrics,plot
