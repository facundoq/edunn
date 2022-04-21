import edunn as nn

from edunn.utils.check_gradient import common_layer
from edunn.utils.check_gradient import cross_entropy_labels, \
    squared_error, \
    binary_cross_entropy_labels


def test_gradients():
    n = 2
    features = 4
    shape = (n, features)
    samples = 100

    ac = nn.AddConstant(4)
    common_layer(ac, shape, samples=samples)

    mc = nn.MultiplyConstant(4)
    common_layer(mc, shape, samples=samples)

    layer = nn.Bias(features, initializer=nn.initializers.RandomUniform())
    common_layer(layer, shape, samples=samples)

    layer = nn.Linear(features, features)
    common_layer(layer, shape, samples=samples)

    layer = nn.Dense(features, features)
    common_layer(layer, shape, samples=samples)

    layer = nn.ReLU()
    common_layer(layer, shape, samples=samples)

    layer = nn.TanH()
    common_layer(layer, shape, samples=samples)

    layer = nn.Sigmoid()
    common_layer(layer, shape, samples=samples)

    layer = nn.Softmax()
    common_layer(layer, shape, samples=samples)

    layer = nn.SquaredError()
    squared_error(layer, shape, samples=samples, tolerance=1e-3)

    layer = nn.BinaryCrossEntropy()
    binary_cross_entropy_labels(layer, 3, samples=samples, tolerance=1e-3)

    layer = nn.CrossEntropyWithLabels()
    cross_entropy_labels(layer, (2, 5), samples=samples, tolerance=1e-3)
