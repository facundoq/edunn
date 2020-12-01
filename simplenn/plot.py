import matplotlib.pyplot as plt
import numpy as np

def plot_history(history,filepath):
    plt.plot(history)
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.savefig(filepath)
    plt.close()


def plot_model_dataset_1d(x,y,model,filepath):
    plt.scatter(x,y)
    y_pred=model.predict(x)
    plt.plot(x,y_pred)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(filepath)
    plt.close()


def plot_model_dataset_2d_classification(x,y,model,filepath,detail=0.1,title=""):
    assert x.shape[1]==2,f"x debe tener solo dos variables de entrada (tiene {x.shape[1]})"

    plt.figure()
    # Plot decision regions
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, detail),
                         np.arange(y_min, y_max, detail))

    Z = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(Z)
    Z = Z.argmax(axis=1)
    title = f"{title}: regiones de cada clase"
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)  # ,  cmap='RdBu')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    # Plot data
    plt.scatter(x[:,0],x[:,1],c=y)
    plt.savefig(filepath)
    plt.close()



def plot_model_dataset_1d_classification(x,y,model,filepath):
    xaxis=np.array(range(x.shape[0]))
    plt.scatter(xaxis,x,c=y)
    xx = np.linspace(x.min(),x.max(),100)
    yy = model.predict(xx)
    plt.plot(xx,yy)
    plt.xlabel("index")
    plt.ylabel("x")
    plt.savefig(filepath)
    plt.close()
