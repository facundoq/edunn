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
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    import edunn as nn
    import numpy as np

    return (nn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clasificación con una red de 2 capas

    También podemos entrenar una red neuronal para clasificar las flores del conjunto de datos de [Iris](https://www.kaggle.com/uciml/iris). Podés probar agregando capas adicionales o cambiando a otro problema como el de los Vinos.
    """)
    return


@app.cell
def _(nn):
    x,y,classes=nn.datasets.load_classification("iris")
    # normalización de los datos
    x = (x-x.mean(axis=0))/x.std(axis=0)
    n, din = x.shape
    # calcular cantidad de clases
    classes = y.max()+1
    print("Tamaños de x e y:", x.shape,y.shape)

    hidden_dim=3
    #Red con dos capas 
    model = nn.Sequential([nn.Dense(din,hidden_dim,activation_name="relu"),
                           nn.Dense(hidden_dim,classes,activation_name="softmax"),
                          ])

    error = nn.MeanError(nn.CrossEntropyWithLabels())
    optimizer = nn.GradientDescent(lr=0.1)

    # Algoritmo de optimización
    trainer = nn.SupervisedTrainer(model, optimizer, error, epochs=1000, batch_size=16)
    history = trainer.train(x, y)
    # nn.plot.plot_history(history,error_name=error.name)


    print("Métricas del modelo:")
    y_pred=model.forward(x)
    y_pred_labels=nn.utils.onehot2labels(y_pred)
    nn.metrics.classification_summary(y,y_pred_labels)
    return


if __name__ == "__main__":
    app.run()
