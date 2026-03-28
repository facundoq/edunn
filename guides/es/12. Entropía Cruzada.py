import marimo

__generated_with = "0.21.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
    # Capa de Entropía Cruzada

    En este ejercicio debés implementar la capa de error `CrossEntropyWithLabels`, que permite calcular el error de un modelo que emite probabilidades en términos de distancias entre distribuciones.

    En este caso, el `WithLabels` indica que la distribución de probabilidad verdadera (obtenida del conjunto de datos) en realidad se codifica con etiquetas. de modo que para un problema de `C=3` clases, si un ejemplo es de clase 2 (contando desde 0), entonces su etiqueta es `2`. Esta es una manera cómoda de especificar que su codificación como distribución de probabilidad sería `[0,0,1]`, es decir, un vector de `3` elementos, donde el elemento `2` (de nuevo, contando desde 0), tiene probabilidad 1 y el resto 0.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método forward

    El método `forward` de la capa `CrossEntropyWithLabels` asume que su entrada `y` es una distribución de probabilidades, es decir, `C` valores positivos que suman 1, donde `C` es la cantidad de clases. Asimismo, `y_true` es una etiqueta que indica cual clase de las `C` es la correcta.

    Por ejemplo, si $y=(0.3,0.4,0.3)$ y $y_{true}=2$ entonces habrá un error considerable, ya que el valor $y_{true}=2$ indicaba que se esperaba la distribución  $y=(0,0,1)$. Entonces, los valores $0.3$ y $0.4$ de las clases 0 y 1 deberían bajar, y el valor $0.3$ de la clase 2 debería subir

    La entropía cruzada cuantifica este error calculando el negativo del logaritmo de la probabilidad de la clase
    correcta ($-ln(y_{y_{true}})$), en este caso, de la clase 2 ($-ln(y_2)$). Entonces,

    $$EntropíaCruzada(y,y_true)=EntropíaCruzada((0.3,0.4,0.3),2)=-ln(0.3)=1.20$$

    Reiteramos, en este caso se eligió el valor $0.3$ porque es el que está en el índice 2 del vector $y$, es decir, otra forma de escribir lo anterior sería:

    $$E(y,y_{true})=-ln(y_{y_{true}})=-ln(0.3)=1.20$$

    La razón por la cual se utiliza la función $-ln(0.3)$ para penalizar es que si para la clase correcta la probabilidad es 1, entonces

    $$-ln(y_{y_{true}})=-ln(1)=-0=0$$

    y  no hay penalización. Caso contrario, la salida de $-ln$ será positiva e indicará un error. De esta forma se logra penalizar que la probabilidad de la clase correcta no llegue a 1. Podemos visualizar esto fácilmente en un gráfico de la función $-ln(x)$:

    <img src="img/cross_entropy.png" width="400">

    Por último, como los valores de $y$ están normalizados, no es necesario penalizar que el resto de las probabilidades sea mayor a 0; si el error lleva a que la probabilidad de la clase correcta a ser 1, entonces el resto va a tener que ser 0. Por este motivo (y otros), la entropía cruzada es una buena combinación con la función softmax para entrenar modelos de clasificación.

    En el caso de un lote de ejemplos, el cálculo es independiente para cada ejemplo.

    Implementá el método `forward` de la clase `CrossEntropyWithLabels`:
    """)
    return


@app.cell
def _(nn, np):
    _y = np.array([[1, 0], [0.5, 0.5], [0.5, 0.5]])
    y_true = np.array([0, 0, 1])
    _layer = nn.CrossEntropyWithLabels()
    E = -np.log(np.array([[1], [0.5], [0.5]]))
    nn.utils.check_same(E, _layer.forward(y_true, _y))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Método backward

    Dado que la derivación de las ecuaciones del método `backward` de la entropía cruzada es un poco larga, te dejamos [este apunte](http://facundoq.github.io/edunn/material/crossentropy_derivative) con la derivación de todos los casos.

    Nuevamente, como este error es por cada ejemplo, entonces los cálculos son independientes en cada fila.

    Implementá el método `backward` de la clase `CrossEntropyWithLabels`:
    """)
    return


@app.cell
def _(nn):
    # number of random values of x and δEδy to generate and test gradients
    samples = 100
    batch_size = 2
    features_in = 3
    features_out = 5
    input_shape = (batch_size, features_in)
    _layer = nn.CrossEntropyWithLabels()
    nn.utils.check_gradient.cross_entropy_labels(_layer, input_shape, samples=samples, tolerance=1e-05)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regresión Logística aplicada a la clasificación de Flores

    Ahora que tenemos todos los elementos, podemos definir y entrenar nuestro primer modelo regresión logística para clasificar las flores del conjunto de datos de [Iris](https://www.kaggle.com/uciml/iris).

    Ahora si, vamos a poder hacerlo con la Entropía Cruzada; si bien en este caso los resultados son similares en términos de accuracy, el modelo tiene un error convexo y entonces es más fácil la optimización.
    """)
    return


@app.cell
def _(nn):
    # Cargar datos con las salidas como etiquetas
    # (nota: las etiquetas de clase comienzan en 0)
    x, _y, classes = nn.datasets.load_classification('iris')
    # normalización de los datos
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    n, din = x.shape
    # calcular cantidad de clases
    classes = _y.max() + 1
    print('Tamaños de x e y:', x.shape, _y.shape)
    model = nn.LogisticRegression(din, classes)
    #Modelo de regresión logística, 
    # tiene `din` dimensiones de entrada (4 para iris)
    # y `classes` de salida `3 para iris`
    error = nn.MeanError(nn.CrossEntropyWithLabels())
    # Error cuadrático medio
    optimizer = nn.GradientDescent(lr=0.1)
    trainer = nn.SupervisedTrainer(model, optimizer, error, epochs=1000, batch_size=32)
    history = trainer.train(x, _y)
    # Algoritmo de optimización
    # nn.plot.plot_history(history, error_name=error.name)
    print('Métricas del modelo:')
    y_pred = model.forward(x)
    y_pred_labels = nn.utils.onehot2labels(y_pred)
    nn.metrics.classification_summary(_y, y_pred_labels)
    return


if __name__ == "__main__":
    app.run()
