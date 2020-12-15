import sys
sys.path.insert(0, '..')

import simplenn as sn
import numpy as np
import matplotlib.pyplot as plt
from simplenn import plot
import datasets
from simplenn import metrics
from pathlib import Path
results_dir = Path("results")
results_dir.mkdir(parents=True,exist_ok=True)

dataset_name="study_2d"
#dataset_name="iris"
x,y,classes = datasets.load(dataset_name)
x -= x.mean(axis=0)
x /= x.std(axis=0)

print("Dataset sizes: ",x.shape,y.shape)
n,din=x.shape
n_classes=y.max()+1

optimizer = sn.GradientDescent(lr=0.1)

error = sn.MeanError(sn.CrossEntropyWithLabels())

layers = [sn.Dense(din,n_classes),
          sn.Softmax()
          ]
model = sn.Sequential(layers)
print(model.summary())

epochs=100
batch_size=6

history = model.fit(x,y,error,epochs,batch_size,optimizer)
plot.plot_history(history, results_dir / f"logistic_regression_{dataset_name}_history.png")

y_pred = model.predict(x)
y_pred_labels = y_pred.argmax(axis=1)

metrics.classification_summary(np.squeeze(y), y_pred_labels)


if din ==2:
    plot.plot_model_dataset_2d_classification(x, y, model, results_dir / f"logistic_regression_   {dataset_name}_model.png")
