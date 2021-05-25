import simplenn as sn

from simplenn.utils.check_gradient import common_layer
from simplenn.utils.check_gradient import cross_entropy_labels, \
    squared_error, \
    binary_cross_entropy_labels


def test_gradients():
    n = 2
    features = 4
    shape = (n, features)
    samples = 100

    ac = sn.AddConstant(4)
    common_layer(ac, shape, samples=samples)

    mc = sn.MultiplyConstant(4)
    common_layer(mc, shape, samples=samples)

    layer = sn.Bias(features, initializer=sn.initializers.RandomUniform())
    common_layer(layer, shape, samples=samples)

    layer = sn.Linear(features, features)
    common_layer(layer, shape, samples=samples)

    layer = sn.Dense(features, features)
    common_layer(layer, shape, samples=samples)

    layer = sn.ReLU()
    common_layer(layer, shape, samples=samples)

    layer = sn.TanH()
    common_layer(layer, shape, samples=samples)

    layer = sn.Sigmoid()
    common_layer(layer, shape, samples=samples)

    layer = sn.Softmax()
    common_layer(layer, shape, samples=samples)

    layer = sn.SquaredError()
    squared_error(layer, shape, samples=samples, max_rel_error=1e-3)

    layer = sn.BinaryCrossEntropy()
    binary_cross_entropy_labels(layer, 3, samples=samples, max_rel_error=1e-3)

    layer = sn.CrossEntropyWithLabels()
    cross_entropy_labels(layer, (2, 5), samples=samples, max_rel_error=1e-3)
