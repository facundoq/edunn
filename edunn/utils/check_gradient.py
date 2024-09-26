import numpy as np
from typing import Tuple, Dict, Callable
import edunn as nn

from edunn.utils.model2function import common_layer_to_function, error_layer_to_function, ParameterSet
from edunn.utils.numerical_gradient import numerical_gradient
from colorama import Fore, Back, Style


def common_layer(layer: nn.Model, x_shape: Tuple, samples: int = 1, tolerance=1e-7, break_on_error=True):
    """
    Generates random inputs for samples of x, and random parameters to check
    forward: calculates numerical gradient
    backward: calculates analytical gradient
    and compares them
    """
    f, df, parameter_shapes = common_layer_to_function(layer)
    shapes = {**parameter_shapes, "x": x_shape}
    input_generators = lambda: {k: np.random.normal(0, 1, shape) for k, shape in shapes.items()}
    y = f(input_generators())
    dE_dy_generator = lambda: np.random.normal(0, 1, y.shape)
    common_layer_random_sample(layer, f, df, input_generators, dE_dy_generator, samples=samples, tolerance=tolerance,
                               break_on_error=break_on_error)


def squared_error(l: nn.SquaredError, y_shape: Tuple, samples: int = 1, tolerance=1e-7, break_on_error=True):
    f, df, parameter_shapes = error_layer_to_function(l)

    df_ignore_dE_dy = lambda x, dE_dy: df(x)
    std = 10

    def input_generators():
        parameters = {k: np.random.normal(0, 1, shape) for k, shape in parameter_shapes.items()}
        y = np.random.normal(0, std, y_shape)
        y_true = np.random.normal(0, std, y_shape)
        return {**parameters, "y": y, "y_true": y_true}

    E = f(input_generators())
    dE_generator = lambda: np.ones(E.shape)

    common_layer_random_sample(l, f, df_ignore_dE_dy, input_generators, dE_generator, samples=samples,
                               tolerance=tolerance, break_on_error=break_on_error)


def cross_entropy_labels(l: nn.CrossEntropyWithLabels, y_shape: Tuple, samples: int = 1, tolerance=1e-7,
                         break_on_error=True):
    f, df, parameter_shapes = error_layer_to_function(l)
    df_ignore_dE_dy = lambda x, dE_dy: df(x)

    def input_generators():
        parameters = {k: np.random.normal(0, 1, shape) for k, shape in parameter_shapes.items()}

        y = np.abs(np.random.normal(0, 1, y_shape))
        y /= y.sum(axis=1, keepdims=True)

        n, classes = y.shape
        y_true = np.random.randint(0, classes, (n,))

        return {**parameters, "y": y, "y_true": y_true}

    E = f(input_generators())
    dE_generator = lambda: np.ones(E.shape)
    common_layer_random_sample(l, f, df_ignore_dE_dy, input_generators, dE_generator, samples=samples,
                               tolerance=tolerance, break_on_error=break_on_error)


def binary_cross_entropy_labels(l: nn.BinaryCrossEntropy, batch_size: int, samples: int = 1, tolerance=1e-7,
                                break_on_error=True):
    f, df, parameter_shapes = error_layer_to_function(l)
    df_ignore_dE_dy = lambda x, dE_dy: df(x)
    sm = nn.Softmax()

    def input_generators():
        parameters = {k: np.random.normal(0, 1, shape) for k, shape in parameter_shapes.items()}
        y = np.random.uniform(0, 1, (batch_size, 1))

        y_true = np.random.randint(0, 1, (batch_size, 1))

        return {**parameters, "y": y, "y_true": y_true}

    E = f(input_generators())
    dE_generator = lambda: np.ones(E.shape)
    common_layer_random_sample(l, f, df_ignore_dE_dy, input_generators, dE_generator, samples=samples,
                               tolerance=tolerance, break_on_error=break_on_error)


def common_layer_random_sample(layer: nn.Model, f, df, input_generator, dE_dy_generator, samples: int = 1, tolerance=1e-7,
                               break_on_error=True):
    checks, errors = 0, 0
    print(f"{Back.LIGHTBLUE_EX}{Fore.BLACK}{layer.name} layer:{Style.RESET_ALL}")

    for i in range(samples):
        inputs = input_generator()
        dE_dy = dE_dy_generator()
        dE_dinputs_analytic = df(inputs, dE_dy=dE_dy)
        sample_checks, sample_errors = check_gradient_numerical(f, inputs, dE_dy, dE_dinputs_analytic,
                                                                tolerance=tolerance, break_on_error=break_on_error)
        errors += sample_errors
        checks += sample_checks
        if errors > 0 and break_on_error:
            break

    if errors == 0:
        status = f"{Back.GREEN}{Fore.BLACK}SUCCESS{Style.RESET_ALL}"
        print(f"{status} {checks} partial derivatives checked ({samples} random input samples)")

    if not break_on_error and errors > 0:
        status = f"{Back.RED}{Fore.BLACK}ERROR{Style.RESET_ALL}"
        print(
            f"{status} {checks} partial derivatives checked, {errors} failed (tolerance {tolerance}, {samples} random input samples)")


def check_gradient_numerical(f: Callable, inputs: ParameterSet, dE_dy: np.ndarray,
                             dE_dinputs_analytic: Dict[str, np.ndarray], tolerance: float, break_on_error: bool):
    errors = 0
    checks = 0
    for k, dE_dk_analytical in dE_dinputs_analytic.items():
        v = inputs[k]

        # numerical
        def fk(x):
            old = inputs[k]
            inputs[k] = x
            y = f(inputs)
            inputs[k] = old
            return y

        dE_dk_numerical = numerical_gradient(fk, v, dE_dy)

        # comparison
        error_count, count, max_error = relative_error_count(dE_dk_analytical, dE_dk_numerical, tolerance=tolerance)
        errors += error_count
        checks += count
        if break_on_error and error_count > 0:
            report_errors(dE_dk_analytical, dE_dk_numerical, k, v, dE_dy, max_error, tolerance, )

    return checks, errors


def report_errors(dE_dk_analytical, dE_dk_numerical, label: str, x: np.ndarray, dE_dy: np.ndarray, max_error: float,
                  tolerance: float):
    message = f"{Back.RED}{Fore.BLACK} ERROR {Style.RESET_ALL}"
    message += f"\ndE_d{label}"
    message += f"\n Relative error (max):{max_error:0.5f} (tolerance: {tolerance})"
    message += f"\n{Style.RESET_ALL}######################## Details: ######################## "
    message += f"\n Input {label}:\n{x}"
    message += f"\n Input dE_dy:\n{dE_dy}"
    message += f"\n dE_d{label} (numerical, automatic):\n{dE_dk_numerical}"
    message += f"\n dE_d{label} (analytic, your implementation):\n{dE_dk_analytical}"

    message += f"\n##########################################################\n{Style.RESET_ALL}"
    print(message)


def relative_error_count(danalytical: np.ndarray, dnumerical: np.ndarray, tolerance=1e-8):
    """
    Check numerical gradient vs analytic gradient
    Count how many partial derivatives have a rel error greater than @tolerance
    """

    dsum = abs(dnumerical) + abs(danalytical)
    ddiff = abs(dnumerical - danalytical)
    rel_error = np.zeros_like(dsum)
    non_zero_indices = dsum != 0
    rel_error[non_zero_indices] = ddiff[non_zero_indices] / dsum[non_zero_indices]

    error_count = (rel_error > tolerance).sum()
    if error_count == 0:
        max_error = 0
    else:
        max_error = rel_error.max()

    return error_count, rel_error.size, max_error
