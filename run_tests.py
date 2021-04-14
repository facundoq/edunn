#!/usr/bin/env python3

from tests.test_gradients import test_gradients
from tests.test_regression import test_linear_regression,test_regression_network
from tests.test_classification import test_logistic_regression,test_classification_network


test_gradients()

test_logistic_regression()
test_classification_network()

test_linear_regression()
test_regression_network()
