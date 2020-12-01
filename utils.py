import numpy as np
from colorama import Fore, Back, Style

def check_same(expected:np.ndarray,calculated:np.ndarray,tol=1e-12):
    if approximate_equals(expected,calculated,tol=tol):
        print(f"{Back.GREEN}{Fore.WHITE}SUCCESS{Style.RESET_ALL} Arrays are equal :) (tolerance {tol})")
    else:
        print(f"{Back.RED}{Fore.WHITE}ERROR{Style.RESET_ALL} Arrays are not equal :( (tolerance {tol})")
        print(f"Expected:\n{expected}")
        print(f"Calculated:\n{calculated}")

def approximate_equals(a:np.ndarray,b:np.ndarray,tol=1e-12):
    return np.all(np.abs(a-b)<tol)