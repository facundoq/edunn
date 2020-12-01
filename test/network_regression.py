import sys
sys.path.insert(0, '..')


import simplenn as sn
import numpy as np
import matplotlib.pyplot as plt
from simplenn import plot
import datasets

from pathlib import Path
results_dir = Path("results")
results_dir.mkdir(parents=True,exist_ok=True)

dataset_name="study_regression_1d"
dataset_name="boston"
dataset_name="white_wine"
dataset_name="red_wine"
x,y = datasets.load(dataset_name)

x = x-x.mean(axis=0)
x /= x.std(axis=0)

print("Dataset sizes: ",x.shape,y.shape)
n,din=x.shape
_,dout=y.shape


optimizer = sn.GradientDescent(lr=0.001)
error = sn.MeanError(sn.SquaredError())
layers = [sn.Dense(din,20),
          sn.TanH(),
          sn.Dense(20,10),
          sn.TanH(),
          sn.Dense(10,dout),]
model = sn.Sequential(layers,error)
history = model.fit(x,y,100,16,optimizer)
print(model.summary())

y_pred = model.predict(x)
rmse = np.sqrt(((y-y_pred)**2).mean(axis=0))
mae = np.abs(y-y_pred).mean(axis=0)

print(f"RMSE={rmse}, MAE={mae}")
plot.plot_history(history, results_dir / f"network_regression_{dataset_name}_history.png")
if din ==1 and dout ==1:
    plot.plot_model_dataset_1d(x, y, model, results_dir / f"network_regression_{dataset_name}_model.png")
