import simplenn as sn
import numpy as np
from test.check_gradient import check_gradient_input
from typing import Dict,Tuple
from test.check_gradient import check_gradient_layer_random_sample,debug_gradient_layer_random_sample
if __name__ == '__main__':
    ## Check AddConstant
    n =2
    features = 4
    shape = (n,features)
    samples=1

    ac = sn.AddConstant(4)
    check_gradient_layer_random_sample(ac,shape, samples=samples)

    mc = sn.MultiplyConstant(4)
    check_gradient_layer_random_sample(mc,shape, samples=samples)

    layer = sn.Bias(features, initializer=sn.initializers.RandomUniform())
    check_gradient_layer_random_sample(layer, shape,samples=samples)

    layer = sn.Linear(features, features)
    check_gradient_layer_random_sample(layer, shape, samples=samples)

    layer = sn.Dense(features, features)
    check_gradient_layer_random_sample(layer, shape, samples=samples)

    layer = sn.ReLU()
    check_gradient_layer_random_sample(layer, shape, samples=samples)

    layer = sn.TanH()
    check_gradient_layer_random_sample(layer, shape, samples=samples)

    layer = sn.Sigmoid()
    check_gradient_layer_random_sample(layer, shape, samples=samples)


    layer = sn.Softmax()
    #check_gradient_layer_random_sample(layer, samples, (1, 3),δEδy=1)
    δEδy = np.array([[0,0,1.0,0]])
    check_gradient_layer_random_sample(layer, shape, samples=samples)
    #debug_gradient_layer_random_sample(layer,samples,(1,4))


