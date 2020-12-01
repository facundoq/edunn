from __future__ import print_function
from builtins import range
import numpy as np
from random import randrange
from typing import Tuple,Dict
import simplenn as sn

from .numerical_gradient import replace_params,df,f,layer_to_functions,numerical_gradient
from colorama import Fore, Back, Style

def check_gradient_layer_random_sample(l:sn.Layer, shape:Tuple, samples:int=1, tolerance=1e-7):
    print(f"{Back.LIGHTBLUE_EX}{Fore.BLACK}{l.name} layer:{Style.RESET_ALL}")
    errors=0
    count=0
    for i in range(samples):
        x = np.random.normal(0,1,shape)
        y = l.forward(x)
        δEδy = np.random.normal(0,1,y.shape)
        sample_errors,sample_count=check_gradient_layer(l, x,δEδy, tolerance)
        errors+=sample_errors
        count+=sample_count
    if errors==0:
        status = f"{Back.GREEN}{Fore.BLACK}SUCCESS{Style.RESET_ALL}"
    else:
        status = f"{Back.RED}{Fore.BLACK} ERROR {Style.RESET_ALL}"

    print(f"{status} {count} partial derivatives checked, {errors} failed (tolerance {tolerance}, {samples} random input samples)")


def report_error(label, error_count, count, error_max, tolerance):
    if error_count==0:
        print(f"{label}: success. All {count} checks were within tolerance ({tolerance})")
    else:
        error_percent=error_count/count*100
        print(f"{label}: error. {error_percent:.1f}% checks were NOT within tolerance ({tolerance}), max error: {error_max:.6f}")

def check_gradient_layer(l:sn.Layer, x:np.ndarray, δEδy:np.ndarray, tolerance:float, p:Dict[str, np.ndarray]=None,verbose=False):
    if not p is None:
        replace_params(l, p)
    errors =0
    total_count=0

    δEδx_analytic,δEδp_analytic,y = df(l,x,δEδy)
    error_count,count,error_max  = check_gradient_input("x",lambda x:f(l, x), x, δEδx_analytic,δEδy, tolerance=tolerance)
    errors+=error_count
    total_count+=count

    if verbose and error_count>0:
        report_error(f"{l.name}:δEδx",error_count,count,error_max,tolerance)

    def fp(l:sn.Layer,x:np.ndarray,k:str,p:np.ndarray):
        old_p=l.get_parameters()[k]
        l.get_parameters()[k][:]=p
        y = l.forward(x)
        l.get_parameters()[k][:]=old_p
        return y

    for k in l.get_parameters():
        p0=l.get_parameters()[k]
        error_count,count,error_max = check_gradient_input(k,lambda p:fp(l, x, k, p), p0, δEδp_analytic[k],δEδy, tolerance=tolerance)
        if verbose and error_count>0:
            report_error(f"{l.name}:δEδ{k}",error_count,count,error_max,tolerance)
        errors+=error_count
        total_count+=count


    return errors,total_count

def check_gradient_input(label,f, x, δEδx_analytical:np.ndarray, δEδy:np.ndarray, h=1e-5, tolerance=1e-8, verbose_ok=False, verbose_error=True):
    """
    Check numerical gradients of f wrt x and compare with the analytic gradient
    Can specify a maximum number of checks so that the gradient direction dimensions are randomly sampled.
    """
    δEδx_numerical = numerical_gradient(f,x,δEδy,h)
    δsum=abs(δEδx_numerical) + abs(δEδx_analytical)
    δdiff=abs(δEδx_numerical - δEδx_analytical)
    rel_error = np.zeros_like(δsum)
    non_zero_indices = δsum!=0
    rel_error[non_zero_indices]=δdiff[non_zero_indices] / δsum[non_zero_indices]

    error_count = (rel_error>tolerance).sum()
    if error_count==0:
        max_error=0
    else:
        max_error=rel_error.max()
    message = f"calculating δEδ{label}"
    message+= f"\n Relative error (max):{max_error:0.5f} (tolerance: {tolerance})"
    message+= f"\n{Style.RESET_ALL}######################## Details: ######################## "
    message+= f"\n Input {label}:\n{x}"
    message+= f"\n Input δEδy:\n{δEδy}"
    message+= f"\n δEδ{label} (numerical, automatic):\n{δEδx_numerical}"
    message+= f"\n δEδ{label} (analytic, your implementation):\n{δEδx_analytical}"

    message+= f"\n##########################################################{Style.RESET_ALL}"


    if error_count>0 and verbose_error:
        print(f"{Fore.RED} Error {message}")
    if verbose_ok and error_count ==0:
        print(f"{Fore.GREEN} Success {message}")

    return error_count,rel_error.size,max_error


def debug_gradient_layer_random_sample(l:sn.Layer, samples:int, shape:Tuple,δEδy=None):
    print(f"Layer {l.name}:")

    for i in range(samples):
        x = np.random.normal(0,1,shape)
        y = l.forward(x)
        if δEδy is None:
            δEδy = np.random.normal(0,1,y.shape)
        # print("x:\n",x)
        # print("y:\n",l.forward(x))
        debug_gradient_layer(l,x,δEδy)


def debug_gradient_layer(l:sn.layer,x:np.ndarray,δEδy:np.ndarray):

    δEδx_analytic,δEδp_analytic,y = df(l,x,δEδy)
    fx,fps=layer_to_functions(l)

    print(f"δEδx")
    debug_gradient(fx, x, δEδx_analytic,δEδy)

    for k in l.get_parameters():
        fp = fps[k]
        p0=l.get_parameters()[k]
        print(f"δEδ{k}")
        debug_gradient(lambda p:fp(p,x), p0, δEδp_analytic[k],δEδy)





def debug_gradient(f, x, δEδx_analytic, δEδy, h=1e-4):
    δEδx_numeric=numerical_gradient(f,x,δEδy,h)
    print(f"Analytic gradient:\n {δEδx_analytic} ")
    print(f"Numerical gradient:\n {δEδx_numeric} ")