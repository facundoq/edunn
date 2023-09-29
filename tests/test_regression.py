
import edunn as nn
import edunn.models.mean_error
from edunn import metrics, datasets
from typing import Callable


class ExperimentConfig:
    def __init__(self,max_mse:float,lr:float=0.001,epochs:int=1000):
        self.max_mse=max_mse
        self.lr=lr
        self.epochs=epochs


def evaluate_regression_model(dataset_name:str, model_generator:Callable, epochs:int, lr:float,id:str):
    x,y = datasets.load_regression(dataset_name)
    x = x-x.mean(axis=0)
    x /= x.std(axis=0)
    n,din=x.shape
    _,dout=y.shape
    model = model_generator(din,dout,id)
    print(f"Testing model {model} on dataset {dataset_name}: {n} samples, {din} features, {dout} output values")
    batch_size=min(16,max(64,n//32))
    batch_size = min(n,batch_size)

    optimizer = nn.GradientDescent(batch_size, epochs, lr)
    error = edunn.models.mean_error.MeanError(nn.SquaredError())

    optimizer.optimize(model,x,y,error,)
    y_pred = model.forward(x)
    mse,mae=metrics.rmse(y,y_pred),metrics.mae(y,y_pred)

    return model,mse,mae


def evaluate_regression_model_datasets(model_generator, datasets_config):

    for i,(dataset_name,config) in enumerate(datasets_config.items()):
        lr,epochs,max_mse = config.lr,config.epochs,config.max_mse
        model,mse,mae=evaluate_regression_model(dataset_name, model_generator, epochs, lr,str(i))
        assert mse<config.max_mse,f"Model {model} achieved {mse} rmse, more than  {max_mse} which is the maximum expected for dataset {dataset_name}."
        print(f"\nRMSE={mse} OK (expected less than {max_mse}).")
    print("All models have satisfactory errors.")

def test_linear_regression():
    config_datasets = {
        "boston":ExperimentConfig(3.5),
        "study1d":ExperimentConfig(1.5,epochs=2000),
        "study2d":ExperimentConfig(1.8,epochs=3000),
        "wine_white":ExperimentConfig(0.7),
        "wine_red":ExperimentConfig(0.6),
        "insurance":ExperimentConfig(4500,lr=1e-4),
        "real_state":ExperimentConfig(6.20),
    }

    def linear_regression(din, dout,id):
        layers = [nn.Linear(din, dout),
                  nn.Bias(dout)
                  ]
        return nn.Sequential(layers, f"linear_regression_{id}")


    evaluate_regression_model_datasets(linear_regression, config_datasets)


def test_regression_network():
    config_datasets = {
        "boston":ExperimentConfig(3.2),
        "study1d":ExperimentConfig(2,epochs=2000),
        "study2d":ExperimentConfig(3,epochs=2000),
        "wine_white":ExperimentConfig(0.65,epochs=100),
        "wine_red":ExperimentConfig(0.55,epochs=100),
        "insurance":ExperimentConfig(4500,lr=1e-6,epochs=500),
        "real_state":ExperimentConfig(6.20),
    }



    def network(din,dout,id):
        layers = [nn.Dense(din, din),
                  nn.ReLU(),
                  nn.Dense(din, dout), ]
        return nn.Sequential(layers, f"network_{id}")


    evaluate_regression_model_datasets(network, config_datasets)