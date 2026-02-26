eps = 1e-12

from .model import Model, Phase, ModelWithParameters, ModelWithoutParameters
from .optimizer import (
    Optimizer,
    SGD,
    MomentumSGD,
    NesterovMomentumSGD,
    RMSpropOptimizer,
    AdamOptimizer,
    SignSGDOptimizer,
)
from .trainers import (
    Trainer,
    SupervisedTrainer,
    RecurrentTrainer,
    # Backward-compatible wrappers (combine optimizer + trainer)
    BatchedGradientOptimizer,
    GradientDescent,
    RMSprop,
    Adam,
    MomentumGD,
    NesterovMomentumGD,
    SignGD,
)
from . import initializers, plot

from .models import *
from . import metrics, datasets


from edunn.models.fake import FakeModel, FakeError
