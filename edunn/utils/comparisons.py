import numpy as np
from colorama import Fore, Back, Style
from numbers import Number

default_tolerance = 1e-12
default_confidence = 0.99

success = f"{Back.GREEN}{Fore.BLACK}SUCCESS :){Style.RESET_ALL}"
failure = f"{Back.RED}{Fore.BLACK}ERROR   :({Style.RESET_ALL}"


def check_same(expected: np.ndarray, actual: np.ndarray, tolerance: Number = default_tolerance, check_shape=True):
    if expected is None:
        print(f"{failure} Expected value is None")
        return
    if actual is None:
        print(f"{failure} Actual value is None")
        return

    if check_shape:
        if not expected.shape == actual.shape:
            print(f"{failure} Shape mismatch:")
            print(f"Expected: {expected.shape}")
            print(f"Actual:   {actual.shape}")
            return
    if approximate_equals(expected, actual, tolerance=tolerance):
        print(f"{success} Arrays are equal (tolerance {tolerance})")
    else:
        print(f"{failure} Arrays are not equal (tolerance {tolerance})")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")


def approximate_equals(a: np.ndarray, b: np.ndarray, tolerance: Number = default_tolerance):
    return np.all(np.abs(a - b) < tolerance)


def check_different(a1: np.ndarray, a2: np.ndarray, tolerance: Number = default_tolerance, check_shape=True):
    if check_shape:
        if not a1.shape == a2.shape:
            print(f"{failure} Shape mismatch:")
            print(f"Expected:   {a1.shape}")
            print(f"Calculated: {a2.shape}")
            return

    if not approximate_equals(a1, a2, tolerance=tolerance):
        print(f"{success} Arrays are different  (tolerance {tolerance})")
    else:
        print(f"{failure} Arrays are not different  (tolerance {tolerance})")
        print(f"First:\n{a1}")
        print(f"Second:\n{a2}")


def check_same_shape(a1: np.ndarray, a2: np.ndarray):
    if a1.shape == a2.shape:
        print(f"{success} Arrays are have the same shape: {a1.shape}")
    else:
        print(f"{failure} Arrays have different shapes.")
        print(f"First:\n{a1.shape}")
        print(f"Second:\n{a2.shape}")


def check_same_float(expected: float, actual: float, title: str, tolerance: Number = default_tolerance):
    if np.abs((expected - actual)) < tolerance:
        print(f"{success} {title} is {expected}  (tolerance {tolerance})")
    else:
        print(f"{failure} {title} is not {expected}  (tolerance {tolerance})")
        print(f"Expected: {expected} ")
        print(f"Actual:   {actual}")
