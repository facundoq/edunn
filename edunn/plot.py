import matplotlib.pyplot as plt
plt.style.use('bmh')

import numpy as np
import edunn as nn


def plot_history(history,error_name="Error",filepath=None):
    plt.figure()
    plt.plot(history)
    plt.xlabel("Epochs")
    plt.ylabel(error_name)
    plt.title("Training history")
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
        plt.close()


def plot_model_dataset_1d(x:np.ndarray, y:np.ndarray, model:nn.Model, filepath=None):
    assert x.shape[0]==y.shape[0],"Input and output arrays must have the same number of samples"
    plt.figure()
    plt.scatter(x,y)
    y_pred=model.forward(x)
    plt.plot(x,y_pred)
    plt.xlabel("x")
    plt.ylabel("y")
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
        plt.close()


def plot_model_dataset_2d_classification(x:np.ndarray, y:np.ndarray, model:nn.Model, filepath=None, detail=0.1, title=""):
    assert x.shape[1]==2,f"x must have only two input variables (received x with {x.shape[1]} dimensions)"

    plt.figure()
    # Plot decision regions
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, detail),
                         np.arange(y_min, y_max, detail))

    Z = np.c_[xx.ravel(), yy.ravel()]

    Z = model.forward(Z)
    Z = Z.argmax(axis=1)
    title = f"{title}: Regiones de cada clase"
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)  # ,  cmap='RdBu')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    # Plot data
    plt.scatter(x[:,0],x[:,1],c=y)
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
        plt.close()




def plot_model_dataset_1d_classification(x,y,model,filepath=None):
    plt.figure()
    xaxis=np.array(range(x.shape[0]))
    plt.scatter(xaxis,x,c=y,label="samples")
    xx = np.linspace(x.min(),x.max(),100)
    yy = model.forward(xx)
    plt.plot(xx,yy,label="decision boundary")
    plt.xlabel("index")
    plt.ylabel("x")
    plt.legend()
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
        plt.close()


def regression1d_predictions(y_true:np.ndarray,y_pred:np.ndarray,filepath=None):
    assert y_true.shape[0] == y_pred.shape[0], "Input and output arrays must have the same number of samples"
    plt.figure()
    plt.plot(range(len(y_true)), y_true, label="true value")
    plt.plot(range(len(y_true)), y_pred, label="prediction")
    plt.xlabel("Sample ID")
    plt.ylabel("Prediction")
    plt.legend()
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
        plt.close()

def plot_activation_function(m:nn.Model, range=(-10, 10), backward=False):
    plot_activation_functions([m],range=range,backward=backward)

def plot_activation_functions(models:[nn.Model], range=(-10, 10), backward=False):
    start,end=range
    x = np.linspace(start,end,100)
    names = [m.__class__.__name__ for m in models]
    fig=plt.figure()
    ax = fig.gca()
    # spines_origin(ax)
    for m in models:
        y = m.forward(x)
        name = m.__class__.__name__
        if backward:
            E = np.ones_like(x)
            y,_ = m.backward(E)
            name = f"{name} derivative"

        plt.plot(x,y,label=name)
    plt.legend()
    names = ", ".join(names)
    if backward:
        title = f"Derivatives of {names}"
    else:
        title = f"Values of {names}"
    plt.title(title)

def spines_origin(ax):
    ax.grid(True)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')