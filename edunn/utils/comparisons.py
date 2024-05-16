import numpy as np
from colorama import Fore, Back, Style

default_tol = 1e-12

success = f"{Back.GREEN}{Fore.BLACK}SUCCESS :){Style.RESET_ALL}"
failure = f"{Back.RED}{Fore.BLACK}ERROR   :({Style.RESET_ALL}"


def check_same(expected: np.ndarray, calculated: np.ndarray, tol=default_tol, check_shape=True):
    if expected is None:
        print(f"{failure} Expected value is None")
        return
    if calculated is None:
        print("{failure} Calculated value is None")
        return

    if check_shape:
        if not expected.shape == calculated.shape:
            print(f"{failure} Shape mismatch:")
            print(f"Expected:   {expected.shape}")
            print(f"Calculated: {calculated.shape}")
            return
    if approximate_equals(expected, calculated, tol=tol):
        print(f"{success} Arrays are equal (tolerance {tol})")
    else:
        print(f"{failure} Arrays are not equal (tolerance {tol})")
        print(f"Expected:\n{expected}")
        print(f"Calculated:\n{calculated}")


def approximate_equals(a: np.ndarray, b: np.ndarray, tol=default_tol):
    return np.all(np.abs(a - b) < tol)


def check_different(a1: np.ndarray, a2: np.ndarray, tol=default_tol, check_shape=True):
    if check_shape:
        if not a1.shape == a2.shape:
            print(f"{failure} Shape mismatch:")
            print(f"Expected:   {a1.shape}")
            print(f"Calculated: {a2.shape}")
            return

    if not approximate_equals(a1, a2, tol=tol):
        print(f"{success} Arrays are different  (tolerance {tol})")
    else:
        print(f"{failure} Arrays are not different  (tolerance {tol})")
        print(f"First:\n{a1}")
        print(f"Second:\n{a2}")


def check_same_shape(a1: np.ndarray, a2: np.ndarray):
    if a1.shape == a2.shape:
        print(f"{success} Arrays are have the same shape: {a1.shape}")
    else:
        print(f"{failure} Arrays have different shapes.")
        print(f"First:\n{a1.shape}")
        print(f"Second:\n{a2.shape}")


def check_same_float(expected: float, calculated: float, title: str, tol=default_tol):
    if np.abs((expected - calculated)) < tol:
        print(f"{success} {title} is {expected}  (tolerance {tol})")
    else:
        print(f"{failure} {title} is not {expected}  (tolerance {tol})")
        print(f"Expected:   {expected} ")
        print(f"Calculated: {calculated}")


def check_mean(a: np.ndarray, mean: float, tol=default_tol):
    check_same_float(a.mean(), mean, "Mean", tol=tol)


def check_std(a: np.ndarray, std: float, tol=default_tol):
    check_same_float(a.std(), std, "Std", tol=tol)
