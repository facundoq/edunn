eps = 1e-12

from .layer import SampleErrorLayer,ErrorLayer,Layer,Phase,MeanError,CommonLayer
from .activations import  Sigmoid,TanH,ReLU,MultiplyConstant,AddConstant,Softmax
from .dense import Linear,Dense,Bias
from .error_layer import SquaredError,CrossEntropyWithLabels,BinaryCrossEntropyWithLabels
from .model import Sequential,Model
from .optimizer import RandomOptimizer,GradientDescent
from . import initializers,measures,plot
