import edunn as nn

from edunn.utils.check_gradient import common_layer
from edunn.utils.check_gradient import cross_entropy_labels, squared_error, binary_cross_entropy_labels


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

    layer = nn.GELU()
    common_layer(layer, shape, samples=samples, tolerance=1e-5)

    layer = nn.BatchNorm(features)
    common_layer(layer, (8, features), samples=samples, tolerance=1e-5)

    layer = nn.Dropout()
    layer.set_phase(nn.Phase.Test)
    common_layer(layer, shape, samples=samples)

    image_shape = (2, 3, 7, 7)

    layer = nn.Flatten()
    common_layer(layer, image_shape, samples=3, tolerance=1e-5)

    layer = nn.MaxPool2d(kernel_size=3, stride=1)
    common_layer(layer, image_shape, samples=3, tolerance=1e-5)

    layer = nn.AvgPool2d(kernel_size=3, stride=1)
    common_layer(layer, image_shape, samples=3, tolerance=1e-5)

    layer = nn.Conv2d(3, 4, kernel_size=(3, 3), stride=2, padding=1)
    common_layer(layer, image_shape, samples=3, tolerance=1e-5)

    layer = nn.Conv2d(3, 4, kernel_size=(3, 3), stride=2)
    common_layer(layer, (2, 3, 6, 6), samples=3, tolerance=1e-5)

    layer = nn.Conv2d(3, 4, kernel_size=(3, 1))
    common_layer(layer, image_shape, samples=3, tolerance=1e-5)

    layer = nn.RNN(input_dim=4, output_dim=3, hidden_dim=6)
    common_layer(layer, (2, 5, 4), samples=3, tolerance=1e-5)

    layer = nn.SquaredError()
    squared_error(layer, shape, samples=samples, tolerance=1e-3)

    layer = nn.BinaryCrossEntropy()
    binary_cross_entropy_labels(layer, 3, samples=samples, tolerance=1e-3)

    layer = nn.CrossEntropyWithLabels()
    cross_entropy_labels(layer, (2, 5), samples=samples, tolerance=1e-3)
