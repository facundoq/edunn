import numpy as np
from typing import Tuple,Dict,Callable
import simplenn as sn


from simplenn.utils.model2function import common_layer_to_function,error_layer_to_function,ParameterSet
from simplenn.utils.numerical_gradient import numerical_gradient
from colorama import Fore, Back, Style



def common_layer(l:sn.ModelWithParameters, x_shape:Tuple, samples:int=1, tolerance=1e-7, break_on_error=True):
    f,df,parameter_shapes = common_layer_to_function(l)
    shapes = {**parameter_shapes,"x":x_shape}
    input_generators = lambda: {k:np.random.normal(0,1,shape) for k,shape in shapes.items()}
    y = f(input_generators())
    δEδy_generator = lambda: np.random.normal(0,1,y.shape)
    common_layer_random_sample(l, f, df, input_generators, δEδy_generator, samples=samples, tolerance=tolerance, break_on_error=break_on_error)

def squared_error(l:sn.SquaredError, y_shape:Tuple, samples:int=1, tolerance=1e-7, break_on_error=True):
    f,df,parameter_shapes = error_layer_to_function(l)

    df_ignore_δEδy = lambda x,δEδy: df(x)
    std=10
    def input_generators():
        parameters = {k:np.random.normal(0,1,shape) for k,shape in parameter_shapes.items()}
        y = np.random.normal(0,std,y_shape)
        y_true = np.random.normal(0,std,y_shape)
        return {**parameters,"y":y,"y_true":y_true}
    E = f(input_generators())
    δE_generator = lambda : np.ones(E.shape)

    common_layer_random_sample(l, f, df_ignore_δEδy, input_generators, δE_generator, samples=samples, tolerance=tolerance, break_on_error=break_on_error)

def cross_entropy_labels(l:sn.CrossEntropyWithLabels, y_shape:Tuple, samples:int=1, tolerance=1e-7, break_on_error=True):
    f,df,parameter_shapes = error_layer_to_function(l)
    df_ignore_δEδy = lambda x,δEδy: df(x)

    def input_generators():
        parameters = {k:np.random.normal(0,1,shape) for k,shape in parameter_shapes.items()}

        y = np.abs(np.random.normal(0,1,y_shape))
        y/=y.sum(axis=1,keepdims=True)


        n,classes = y.shape
        y_true=np.random.randint(0,classes,(n,))

        return {**parameters,"y":y,"y_true":y_true}

    E = f(input_generators())
    δE_generator = lambda : np.ones(E.shape)
    common_layer_random_sample(l, f, df_ignore_δEδy, input_generators, δE_generator, samples=samples, tolerance=tolerance, break_on_error=break_on_error)

def binary_cross_entropy_labels(l:sn.BinaryCrossEntropy, batch_size:int, samples:int=1, tolerance=1e-7, break_on_error=True):
    f,df,parameter_shapes = error_layer_to_function(l)
    df_ignore_δEδy = lambda x,δEδy: df(x)
    sm=sn.Softmax()
    def input_generators():
        parameters = {k:np.random.normal(0,1,shape) for k,shape in parameter_shapes.items()}
        y = np.random.uniform(0,1,(batch_size,1))

        y_true=np.random.randint(0,1,(batch_size,1))

        return {**parameters,"y":y,"y_true":y_true}

    E = f(input_generators())
    δE_generator = lambda : np.ones(E.shape)
    common_layer_random_sample(l, f, df_ignore_δEδy, input_generators, δE_generator, samples=samples, tolerance=tolerance, break_on_error=break_on_error)


def common_layer_random_sample(l:sn.Model, f, df, input_generator, δEδy_generator, samples:int=1, tolerance=1e-7, break_on_error=True):
    checks,errors=0,0
    print(f"{Back.LIGHTBLUE_EX}{Fore.BLACK}{l.name} layer:{Style.RESET_ALL}")

    for i in range(samples):
        inputs = input_generator()
        δEδy = δEδy_generator()
        δEδinputs_analytic = df(inputs,δEδy=δEδy)
        sample_checks,sample_errors=check_gradient_numerical(f,inputs,δEδy,δEδinputs_analytic,tolerance=tolerance,break_on_error=break_on_error)
        errors+=sample_errors
        checks+=sample_checks
        if errors>0 and break_on_error:
            break

    if errors==0:
        status = f"{Back.GREEN}{Fore.BLACK}SUCCESS{Style.RESET_ALL}"
        print(f"{status} {checks} partial derivatives checked ({samples} random input samples)")

    if not break_on_error and errors>0:
        status = f"{Back.RED}{Fore.BLACK}ERROR{Style.RESET_ALL}"
        print(f"{status} {checks} partial derivatives checked, {errors} failed (tolerance {tolerance}, {samples} random input samples)")



def check_gradient_numerical(f:Callable,inputs:ParameterSet,δEδy:np.ndarray,δEδinputs_analytic:Dict[str,np.ndarray], tolerance:float,break_on_error:bool):
    errors=0
    checks=0
    for k,δEδk_analytical in δEδinputs_analytic.items():
        v  = inputs[k]
        #numerical
        def fk(x):
            old =inputs[k]
            inputs[k] = x
            y = f(inputs)
            inputs[k] = old
            return y
        δEδk_numerical = numerical_gradient(fk,v,δEδy)

        #comparison
        error_count,count,max_error=relative_error_count(δEδk_analytical,δEδk_numerical,tolerance=tolerance)
        errors+=error_count
        checks+= count
        if break_on_error and error_count>0 :

            report_errors(δEδk_analytical,δEδk_numerical,k,v,δEδy,max_error,tolerance,)

    return checks,errors


def report_errors(δEδk_analytical,δEδk_numerical,label:str,x:np.ndarray,δEδy:np.ndarray,max_error:float,tolerance:float):

    message = f"{Back.RED}{Fore.BLACK} ERROR {Style.RESET_ALL}"
    message+= f"\nδEδ{label}"
    message+= f"\n Relative error (max):{max_error:0.5f} (tolerance: {tolerance})"
    message+= f"\n{Style.RESET_ALL}######################## Details: ######################## "
    message+= f"\n Input {label}:\n{x}"
    message+= f"\n Input δEδy:\n{δEδy}"
    message+= f"\n δEδ{label} (numerical, automatic):\n{δEδk_numerical}"
    message+= f"\n δEδ{label} (analytic, your implementation):\n{δEδk_analytical}"

    message+= f"\n##########################################################\n{Style.RESET_ALL}"
    print(message)

def relative_error_count(δanalytical:np.ndarray, δnumerical:np.ndarray, tolerance=1e-8):
    """
    Check numerical gradient vs analytic gradient
    Count how many partial derivatives have a rel error greater than @tolerance
    """

    δsum= abs(δnumerical) + abs(δanalytical)
    δdiff=abs(δnumerical - δanalytical)
    rel_error = np.zeros_like(δsum)
    non_zero_indices = δsum!=0
    rel_error[non_zero_indices]=δdiff[non_zero_indices] / δsum[non_zero_indices]

    error_count = (rel_error>tolerance).sum()
    if error_count==0:
        max_error=0
    else:
        max_error=rel_error.max()

    return error_count,rel_error.size,max_error
