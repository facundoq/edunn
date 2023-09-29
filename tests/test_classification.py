import sys

import edunn.models.mean_error

sys.path.insert(0, '..')
import edunn as nn
import numpy as np
from edunn import metrics, datasets
from typing import Callable


class ExperimentConfig:
    def __init__(self,min_accuracy:float,lr:float=0.01,epochs:int=1000):
        self.min_accuracy=min_accuracy
        self.lr=lr
        self.epochs=epochs

def evaluate_classification_model(dataset_name:str, model_generator:Callable, epochs:int, lr:float,id:str):
    x,y,classes = datasets.load_classification(dataset_name)
    x = x-x.mean(axis=0)
    x /= x.std(axis=0)
    n,din=x.shape
    assert (n == len(y))
    n_classes =len(classes)
    model = model_generator(din,n_classes,id)
    print(f"Testing model {model} on dataset {dataset_name}: {n} samples, {din} features, {n_classes} classes")
    batch_size=min(16,max(64,n//32))
    batch_size = min(n,batch_size)
    optimizer = nn.GradientDescent(batch_size, epochs//10, lr)

    if n_classes==2:
        sample_error=nn.BinaryCrossEntropy()
    else:
        sample_error=nn.CrossEntropyWithLabels()

    error = edunn.models.mean_error.MeanError(sample_error)
    optimizer.optimize(model,x,y,error)
    y_pred = model.forward(x)
    y_pred_labels= np.argmax(y_pred,axis=1)
    accuracy=metrics.accuracy(y,y_pred_labels)

    return model,accuracy


def evaluate_classification_model_datasets(model_generator, datasets_config):

    for i,(dataset_name,config) in datasets_config.items():
        lr,epochs,min_accuracy = config.lr,config.epochs,config.min_accuracy
        model,accuracy=evaluate_classification_model(dataset_name, model_generator, epochs, lr,str(i))
        assert min_accuracy<=accuracy,f"Model {model} achieved {accuracy} accuracy, less that  {min_accuracy} which is the expected for dataset {dataset_name}"
        print(f"Accuracy={accuracy} OK (expected more than {min_accuracy})\n")
    print("All models have satisfactory accuracies.\n")

def test_logistic_regression():
    config_datasets = {
        "study1d":ExperimentConfig(0.5),
        "study2d_easy":ExperimentConfig(0.5),
        "study2d":ExperimentConfig(0.5),
        "iris":ExperimentConfig(0.95,epochs=2000),
    }

    def logistic_regression(din, classes,id):
        if classes==2:
            classes=1
            last_layer=nn.Sigmoid()
        else:
            last_layer=nn.Softmax()

        layers = [nn.Linear(din, classes),
                  nn.Bias(classes),
                  last_layer,
                  ]
        return nn.Sequential(layers, f"logistic_regression_{id}")

    evaluate_classification_model_datasets(logistic_regression, config_datasets)

def test_classification_network():
    config_datasets = {
        "study1d":ExperimentConfig(0.5),
        "study2d_easy":ExperimentConfig(0.5),
        "study2d":ExperimentConfig(0.5),
        "iris":ExperimentConfig(0.95,lr=0.1),
    }


    def network(din,classes,id):

        if classes==2:
            classes=1
            last_layer=nn.Sigmoid()
        else:
            last_layer=nn.Softmax()

        layers = [nn.Dense(din, din),
                  nn.ReLU(),
                  nn.Dense(din, classes),
                  last_layer,
                  ]
        return nn.Sequential(layers, f"network_{id}")


    evaluate_classification_model_datasets(network, config_datasets)