import simplenn as sn
import numpy as np
from typing import Dict,Tuple
from test.check_gradient import check_gradient_common_layer,check_gradient_cross_entropy_labels,check_gradient_squared_error,check_gradient_binary_cross_entropy_labels
if __name__ == '__main__':
    ## Check AddConstant
    n =2
    features = 4
    shape = (n,features)
    samples=100

    ac = sn.AddConstant(4)
    check_gradient_common_layer(ac,shape, samples=samples)

    mc = sn.MultiplyConstant(4)
    check_gradient_common_layer(mc,shape, samples=samples)

    layer = sn.Bias(features, initializer=sn.initializers.RandomUniform())
    check_gradient_common_layer(layer, shape,samples=samples)

    layer = sn.Linear(features, features)
    check_gradient_common_layer(layer, shape, samples=samples)

    layer = sn.Dense(features, features)
    check_gradient_common_layer(layer, shape, samples=samples)

    layer = sn.ReLU()
    check_gradient_common_layer(layer, shape, samples=samples)

    layer = sn.TanH()
    check_gradient_common_layer(layer, shape, samples=samples)

    layer = sn.Sigmoid()
    check_gradient_common_layer(layer, shape, samples=samples)

    layer = sn.Softmax()
    check_gradient_common_layer(layer, shape, samples=samples)

    layer = sn.CrossEntropyWithLabels()
    check_gradient_cross_entropy_labels(layer, (2,5), samples=samples, max_rel_error=1e-3)

    layer = sn.SquaredError()
    check_gradient_squared_error(layer, shape, samples=samples, max_rel_error=1e-3)

    layer = sn.BinaryCrossEntropyWithLabels()
    check_gradient_binary_cross_entropy_labels(layer, 3, samples=samples, max_rel_error=1e-3)