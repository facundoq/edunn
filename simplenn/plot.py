import matplotlib.pyplot as plt
import numpy as np
import simplenn as sn

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


def plot_model_dataset_1d(x:np.ndarray,y:np.ndarray,model:sn.Model,filepath=None):
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


def plot_model_dataset_2d_classification(x:np.ndarray,y:np.ndarray,model:sn.Model,filepath=None,detail=0.1,title=""):
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
    plt.legend()
    if filepath is None:
        plt.show()
    else:
        plt.savefig(filepath)
        plt.close()
