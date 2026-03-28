import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import sys
    sys.path.insert(0, '../..')
    return (sys,)


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    import numpy as np
    import edunn as nn
    from edunn import utils

    return nn, np, utils


@app.cell
def _(np, sys):
    np.set_printoptions(threshold=sys.maxsize)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # RMSprop
    """)
    return


@app.cell
def _(nn, np, utils):
    #Modelo falso con un vector de parámetros con valor inicial [0,0] y gradientes que siempre son [1,-11]
    model = nn.FakeModel(parameter=np.array([0, 0]), gradient=np.array([1, -1]))
    # función de error falso cuyo error es siempre 1 y las derivadas también
    error = nn.FakeError(error=1, derivative_value=1)
    _fake_samples = 3
    # Conjunto de datos falso, que no se utilizará realmente
    _fake_x = np.random.rand(_fake_samples, 10)
    _fake_y = np.random.rand(_fake_samples, 5)
    optimizer = nn.RMSprop(lr=2)
    _trainer = nn.SupervisedTrainer(model, optimizer, error, verbose=False, epochs=1, batch_size=_fake_samples)
    # Optimizar el modelo por 1 época con lr=2
    _history = _trainer.train(_fake_x, _fake_y)
    _expected_parameters = np.array([-19, 19])
    utils.check_same(_expected_parameters, model.get_parameters()['parameter'])
    _trainer = nn.SupervisedTrainer(model, optimizer, error, verbose=False, epochs=100, batch_size=32)
    _history = _trainer.train(_fake_x, _fake_y)
    _expected_parameters = np.array([-33, 33])
    # Optimizar el modelo por 1 época *adicional* con lr=2
    utils.check_same(_expected_parameters, model.get_parameters()['parameter'])
    optimizer = nn.RMSprop(lr=1)
    _trainer = nn.SupervisedTrainer(model, optimizer, error, verbose=False, epochs=3, batch_size=_fake_samples)
    _history = _trainer.train(_fake_x, _fake_y)
    _expected_parameters = np.array([-54, 54])
    # Optimizar el modelo por 3 épocas más, ahora con con lr=1    
    utils.check_same(_expected_parameters, model.get_parameters()['parameter'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Adam
    """)
    return


@app.cell
def _(nn, np, utils):
    model_1 = nn.FakeModel(parameter=np.array([0, 0]), gradient=np.array([1, -1]))
    error_1 = nn.FakeError(error=1, derivative_value=1)
    _fake_samples = 3
    _fake_x = np.random.rand(_fake_samples, 10)
    _fake_y = np.random.rand(_fake_samples, 5)
    optimizer_1 = nn.Adam(lr=2)
    _trainer = nn.SupervisedTrainer(model_1, optimizer_1, error_1, verbose=False, epochs=1, batch_size=_fake_samples)
    _history = _trainer.train(_fake_x, _fake_y)
    _expected_parameters = np.array([-1, 1])
    utils.check_same(_expected_parameters, model_1.get_parameters()['parameter'])
    _trainer = nn.SupervisedTrainer(model_1, optimizer_1, error_1, verbose=False, epochs=100, batch_size=32)
    _history = _trainer.train(_fake_x, _fake_y)
    _expected_parameters = np.array([-3, 3])
    utils.check_same(_expected_parameters, model_1.get_parameters()['parameter'])
    optimizer_1 = nn.Adam(lr=1)
    _trainer = nn.SupervisedTrainer(model_1, optimizer_1, error_1, verbose=False, epochs=3, batch_size=_fake_samples)
    _history = _trainer.train(_fake_x, _fake_y)
    _expected_parameters = np.array([-5, 5])
    utils.check_same(_expected_parameters, model_1.get_parameters()['parameter'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comprobaciones mediante entrenamiento
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RMSprop
    """)
    return


@app.cell
def _(classes, din, nn):
    _hidden_dim = 32
    model_2 = nn.Sequential([nn.Dense(din, _hidden_dim, activation_name='relu'), nn.Dense(_hidden_dim, classes, activation_name='softmax')])
    error_2 = nn.MeanError(nn.CrossEntropyWithLabels())
    optimizer_2 = nn.RMSprop(lr=0.01)
    return error_2, model_2, optimizer_2


@app.cell
def _(error_2, model_2, nn, optimizer_2, x, y):
    _trainer = nn.SupervisedTrainer(model_2, optimizer_2, error_2, epochs=100, batch_size=32)
    _history = _trainer.train(x, y)
    # nn.plot.plot_history(_history, error_name=error_2.name)
    return


@app.cell
def _(model_2, nn, x, y):
    print('Métricas del modelo:')
    _y_pred = model_2.forward(x)
    _y_pred_labels = nn.utils.onehot2labels(_y_pred)
    nn.metrics.classification_summary(y, _y_pred_labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adam
    """)
    return


@app.cell
def _(classes, din, nn):
    _hidden_dim = 32
    model_3 = nn.Sequential([nn.Dense(din, _hidden_dim, activation_name='relu'), nn.Dense(_hidden_dim, classes, activation_name='softmax')])
    error_3 = nn.MeanError(nn.CrossEntropyWithLabels())
    optimizer_3 = nn.Adam(lr=0.01)
    return error_3, model_3, optimizer_3


@app.cell
def _(error_3, model_3, nn, optimizer_3, x, y):
    _trainer = nn.SupervisedTrainer(model_3, optimizer_3, error_3, epochs=100, batch_size=32)
    _history = _trainer.train(x, y)
    # nn.plot.plot_history(_history, error_name=error_3.name)
    return


@app.cell
def _(model_3, nn, x, y):
    print('Métricas del modelo:')
    _y_pred = model_3.forward(x)
    _y_pred_labels = nn.utils.onehot2labels(_y_pred)
    nn.metrics.classification_summary(y, _y_pred_labels)
    return


if __name__ == "__main__":
    app.run()
