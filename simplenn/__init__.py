eps = 1e-12

from .model import ErrorModel,Model,Phase, ModelWithParameters
from .optimizer import Optimizer,BatchedGradientOptimizer,GradientDescent,MomentumGD,NesterovMomentumGD,SignGD
from . import initializers,plot

from .models import *
from . import metrics,datasets


from simplenn.models.fake import FakeModel,FakeError
