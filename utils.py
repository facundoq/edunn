import numpy as np
from colorama import Fore, Back, Style
default_tol=1e-12

def check_same(expected:np.ndarray,calculated:np.ndarray,tol=default_tol,check_shape=True):
    if check_shape:
        if not expected.shape ==calculated.shape:
            print(f"Shape mismatch:")
            print(f"Expected:   {expected.shape}")
            print(f"Calculated: {calculated.shape}")
            return
    if approximate_equals(expected,calculated,tol=tol):
        print(f"{Back.GREEN}{Fore.BLACK}SUCCESS{Style.RESET_ALL} Arrays are equal :) (tolerance {tol})")
    else:
        print(f"{Back.RED}{Fore.BLACK}ERROR{Style.RESET_ALL} Arrays are not equal :( (tolerance {tol})")
        print(f"Expected:\n{expected}")
        print(f"Calculated:\n{calculated}")

def approximate_equals(a:np.ndarray,b:np.ndarray,tol=default_tol):
    return np.all(np.abs(a-b)<tol)



def check_different(a1:np.ndarray, a2:np.ndarray, tol=default_tol):
    if not approximate_equals(a1, a2, tol=tol):
        print(f"{Back.GREEN}{Fore.BLACK}SUCCESS{Style.RESET_ALL} Arrays are different :) (tolerance {tol})")
    else:
        print(f"{Back.RED}{Fore.BLACK}ERROR{Style.RESET_ALL} Arrays are not different :( (tolerance {tol})")
        print(f"First:\n{a1}")
        print(f"Second:\n{a2}")


def check_same_float(expected:float,calculated:float,title:str,tol=default_tol):
    if np.abs((expected-calculated))<tol:
        print(f"{Back.GREEN}{Fore.BLACK}{Style.DIM}SUCCESS{Style.RESET_ALL} {title} is {expected} :) (tolerance {tol})")
    else:
        print(f"{Back.RED}{Fore.BLACK}ERROR{Style.RESET_ALL} {title} is not {expected} :( (tolerance {tol})")
        print(f"Expected:   {expected} ")
        print(f"Calculated: {calculated}")

def check_mean(a:np.ndarray, mean:float, tol=default_tol):
    check_same_float(a.mean(),mean,"Mean",tol=tol)

def check_std(a:np.ndarray, std:float, tol=default_tol):
    check_same_float(a.std(),std,"Std",tol=tol)