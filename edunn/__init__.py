eps = 1e-12

from .model import Model,Phase,ModelWithParameters, ModelWithoutParameters
from .optimizer import Optimizer,BatchedGradientOptimizer,GradientDescent,MomentumGD,NesterovMomentumGD,SignGD
from . import initializers,plot

from .models import *
from . import metrics,datasets


from edunn.models.fake import FakeModel,FakeError