import numpy as np
import numpy.testing as testing
import pytest

from edunn.utils.numerical_gradient import numerical_gradient

testdata = [
    (lambda x: np.array([42, x[0], x[1] * x[1], np.log(x[2])]), np.ones(3), np.ones(4), np.array([1, 2, 1])),
]


@pytest.mark.parametrize("f, x, dE_dy, expected", testdata)
def test_numerical_gradient_constant(f, x, dE_dy, expected):
    gradient = numerical_gradient(f, x, dE_dy)

    testing.assert_allclose(actual=gradient, desired=expected, atol=1e-5)
