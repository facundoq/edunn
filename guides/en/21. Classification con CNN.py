import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note:** This guide is currently in Spanish. An English translation is pending.
    """)
    return


@app.cell
def _():
    import sys
    sys.path.insert(0, '../..')
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    import edunn as nn
    import numpy as np

    return nn, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clasificación con CNN

    También podemos entrenar una red neuronal para clasificar las imágenes de dígitos escritos a mano del conjunto de datos de [MNIST](http://yann.lecun.com/exdb/mnist/). Este conjunto de datos es un clásico en el aprendizaje automático, y es un buen punto de partida para probar las redes neuronales convolucionales. Intentá probar agregando/quitando capas adicionales y variando los parámetros de las mismas.
    """)
    return


@app.cell
def _(nn):
    x,y,classes=nn.datasets.load_classification("mnist")
    # normalización de los datos
    x = (x-x.mean())/x.std()
    n, din = x.shape
    # calcular cantidad de clases
    classes = y.max()+1
    print("Tamaños de x e y:", x.shape,y.shape)
    x.min(), x.max()
    return classes, din, x, y


@app.cell
def _(np, x):
    i=np.random.randint(0, x.shape[0])
    return (i,)


@app.cell
def _(i, np, x, y):
    import matplotlib.pyplot as plt
    plt.title(y[i])
    plt.imshow(np.reshape(x[i],(28,28)),cmap="gray")
    return


@app.cell
def _(x):
    x_1 = x.reshape(-1, 1, 28, 28)
    return (x_1,)


@app.cell
def _(x_1):
    def calculate_in_features(input_size, layers):
        output_size = input_size
        for layer in layers:
            output_size = (output_size - layer['kernel_size'] + 2 * layer['padding']) // layer['stride'] + 1
            out_channels = layer['out_channels']
        in_features = output_size * output_size * out_channels
        return in_features
    layers = [{'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0}, {'out_channels': 32, 'kernel_size': 2, 'stride': 2, 'padding': 0}]
    input_size = x_1.shape[-1]
    in_features = calculate_in_features(input_size, layers)
    in_features
    return (in_features,)


@app.cell
def _(classes, in_features, nn):
    #Red convolucional
    initializer = nn.initializers.KaimingNormal()#nn.initializers.RandomNormal(1e-20)
    model = nn.Sequential([
        nn.Conv2d(in_channels=1,     out_channels=32,    kernel_size=(3,3),      stride=1, padding=0, kernel_initializer=initializer),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Dense(input_size=in_features,output_size=32,activation_name="relu",linear_initializer=None),
        nn.Dense(input_size=32,output_size=classes,activation_name="softmax",linear_initializer=None),
        ])

    error = nn.MeanError(nn.CrossEntropyWithLabels())
    optimizer = nn.GradientDescent(lr=0.1)
    return error, model, optimizer


@app.cell
def _():
    # np.set_printoptions(threshold=sys.maxsize)
    return


@app.cell
def _(error, model, nn, optimizer, x_1, y):
    _trainer = nn.SupervisedTrainer(model, optimizer, error, epochs=100, batch_size=32)
    _history = _trainer.train(x_1, y)
    # nn.plot.plot_history(_history, error_name=error.name)
    return


@app.cell
def _(model, nn, x_1, y):
    print('Métricas del modelo:')
    _y_pred = model.forward(x_1)
    _y_pred_labels = nn.utils.onehot2labels(_y_pred)
    nn.metrics.classification_summary(y, _y_pred_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(x_1):
    x_2 = x_1.squeeze().reshape(-1, 28 * 28)
    return (x_2,)


@app.cell
def _(classes, din, nn):
    hidden_dim = 32
    #Red con dos capas 
    model_1 = nn.Sequential([nn.Dense(din, hidden_dim, activation_name='relu'), nn.Dense(hidden_dim, classes, activation_name='softmax')])
    error_1 = nn.MeanError(nn.CrossEntropyWithLabels())
    optimizer_1 = nn.GradientDescent(lr=0.01)
    return error_1, model_1, optimizer_1


@app.cell
def _(error_1, model_1, nn, optimizer_1, x_2, y):
    _trainer = nn.SupervisedTrainer(model_1, optimizer_1, error_1, epochs=100, batch_size=32)
    _history = _trainer.train(x_2, y)
    # nn.plot.plot_history(_history, error_name=error_1.name)
    return


@app.cell
def _(model_1, nn, x_2, y):
    print('Métricas del modelo:')
    _y_pred = model_1.forward(x_2)
    _y_pred_labels = nn.utils.onehot2labels(_y_pred)
    nn.metrics.classification_summary(y, _y_pred_labels)
    return


if __name__ == "__main__":
    app.run()
