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
    # Librería **edunn**

    [edunn](https://github.com/facundoq/edunn) es una librería para definir y entrenar redes neuronales basada en [Numpy](https://numpy.org/), diseñada para ser simple de _entender_.

    Aún más importante, fue diseñada para que sea simple de _implementar_. Es decir, su uso **principal** es como objeto de aprendizaje para comprender como se implementan las redes neuronales modernas en frameworks como [Keras](https://keras.io/) o [Pytorch](https://pytorch.org/).

    No obstante, también es simple para _utilizar_. Por ejemplo, para definir y entrenar una red neuronal para clasificación con de tres capas con distintas funciones de activación, podemos escribir un código muy similar al de estos frameworks:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```python
    import edunn as nn

    dataset_name="iris"
    x,y,classes = nn.datasets.load_classification(dataset_name)
    n,din=x.shape
    n_classes=y.max()+1

    # Definición del modelo
    layers = [nn.Linear(din,10),
              nn.Bias(10),
              nn.ReLU(),

              nn.Linear(10,n_classes),
              nn.Bias(n_classes),
              nn.Softmax()
              ]

    model = nn.Sequential(layers)
    print("Arquitectura de la Red:")
    print(model.summary())

    error = nn.MeanError(nn.CrossEntropyWithLabels())
    # Algoritmo de optimización
    optimizer = nn.GradientDescent(lr=0.1)

    # Algoritmo de optimización
    print("Entrenando red con descenso de gradiente:")
    history = optimizer.optimize(model,x,y,error)

    # Reporte del desempeño
    y_pred = model.forward(x)
    y_pred_labels = y_pred.argmax(axis=1)
    print(f"Accuracy final del modelo en el conjunto de entrenamiento: {nn.metrics.accuracy(y,y_pred_labels)*100:0.2f}%")

    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conocimiento previo
    Para poder implementar la librería, asumimos que ya has adquirido los conceptos básicos de redes neuronales:

    * Capas
        * Capas Lineales
        * Funciones de Activación
        * Composición de capas
        * Métodos forward y backward
    * Algoritmo de propagación hacia atrás (backpropagation)
    * Descenso de gradiente
        * Cálculo de gradientes
        * Optimización básica por gradientes
    * Cómputo/entrenamiento por lotes (batches)

    También se asume conocimiento de Python y de Numpy, así como del manejo de bases de datos tabulares y de imágenes.

    # Componentes de la librería

    Describimos los componentes básicos de la librería utilizados en el código anterior, para proveer el contexto de los ejercicios a realizar.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Módulo **datasets**

    El módulo `edunn.datasets` que permite cargar algunos conjuntos de datos de prueba fácilmente. Estos conjuntos de datos se utilizarán para verificar y experimentar con los modelos.
    """)
    return


@app.cell
def _():
    import edunn as nn
    dataset_name="study2d"
    x,y,classes = nn.datasets.load_classification(dataset_name)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    n,din=x.shape
    n_classes=y.max()+1

    print(f"El conjunto de datos {dataset_name} tiene {n} ejemplos, {din} características por ejemplo y {n_classes} clases: {classes}.")
    return dataset_name, din, n_classes, nn, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Para ver qué otros conjuntos de datos para clasificación o regresión tiene módulo  `datasets` de `edunn` (que accedemos como `nn.datasets`), podés ejecutar `nn.datasets.get_classification_names()` y `nn.datasets.get_regression_names()` y obtener una lista de nombres.
    """)
    return


@app.cell
def _(nn):
    print("Los conjuntos de datos de clasificación disponibles son:")
    print(nn.datasets.get_classification_names())
    print()

    print("Los conjuntos de datos de regresión disponibles son:")
    print(nn.datasets.get_regression_names())
    print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clases y módulos de edunn

    Para usar `edunn`, importamos la librería y la llamamos `nn` de modo que sea más fácil de tipear.
    ```python
    import edunn as nn
    ```

    La librería tiene una clase fundamental, `Model`. Esta es la superclase de los modelos/capas que implementaremos, y define dos métodos abstractos para que implementen sus subclases:

    * `forward(x)`: computa la salida `y` dada una entrada `x`.
        * Como asunción para simplificar la librería, los modelos sólo podrán tener una entrada y una salida, que deben ser un arreglo de numpy (excepto los de error). En la práctica, veremos que esta no es una limitación importante.
    * `backward(δEδy)`: computa el gradiente del error respecto a la entrada (`δEδx`), utilizando el gradiente del error respecto a la salida (`δEδy`). Si el modelo/capa tiene parámetros, también calcula el gradiente respecto a estos parámetros.
        * `backward` permite hacer una implementación desacoplada del algoritmo backpropagation.
        * Utilizando el `backward` de un modelo, se puede optimizarlo mediante descenso de gradiente.

    La librería tiene varias clases de distintas capas/modelos:

    * Las clases `Linear`, `Bias`, que permiten crear capas con las funciones $wx$ y $x+b$ respectivamente. En estos casos, $w$ y $b$ son parámetros a optimizar. Combinando estas capas se puede formar una capa densa tradicional que calcula $wx+b$.
    * Las clases `TanH`, `ReLU` y `Softmax`, que permiten crear capas con las funciones de activación de esos nombres.
    * La clase `Sequential` para crear redes secuenciales, donde donde la salida de cada capa es la entrada de la capa siguiente, y hay solo una capa inicial y una final.

    Cada una de estas clases es una subclase de `Model`, y por ende permite hacer las 2 operaciones fundamentales, `forward` y `backward`. En adelante, usaremos la palabra _capa_ como sinónimo de modelo, es decir, de una subclase de `Model`. Esta terminología, si bien es un poco inexacta, es estándar en el campo de las redes neuronales.
    """)
    return


@app.cell
def _(din, n_classes, nn):
    layers = [nn.Linear(din,10),
              nn.Bias(10),
              nn.ReLU(),
              nn.Linear(10,20),
              nn.Bias(20),
              nn.TanH(),
              nn.Linear(20,n_classes),
              nn.Bias(n_classes),
              nn.Softmax()
              ]
    model = nn.Sequential(layers)
    print("Resumen del modelo:")
    print(model.summary())
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Capas/Modelos de error

    Los modelos requieren medir su error. Para eso `edunn` también tiene algunas capas/modelos de error, que reciben en su `forward` dos entradas, la calculada por la red y la esperada. Tenemos dos tipos de capas:

    * Aquellas que permiten calcular el error de la red _para cada ejemplo por separado_, como `CrossEntropyWithLabels`, o `SquaredError`
    * Aquellas que permiten combinar los errores de cada ejemplo para generar un error que sea escalar.
        * La capa `MeanError`  permite calcular el error promedio de otra capa de error, como la capa `CrossEntropyWithLabels` y `SquaredError` que mencionamos.
    """)
    return


@app.cell
def _(nn):
    mean_cross_entropy_error = nn.MeanError(nn.CrossEntropyWithLabels())
    mean_squared_error = nn.MeanError(nn.SquaredError())
    return (mean_cross_entropy_error,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Optimizadores

    Para entrenar un modelo, podemos utilizar un objeto `Optimizer`, cuyo método `optimize` permite, dados arreglos `x` e `y` y una función de error, entrenar un modelo para minimizar ese error en este conjunto de datos.
    Para este entrenamiento, debe especificarse un algoritmo de optimización. En este caso utilizamos descenso de gradiente simple con la clase `GradientDescent`, una tasa de aprendizaje de `0.1`, `100` épocas y un tamaño de lote de 8.
    """)
    return


@app.cell
def _(mean_cross_entropy_error, model, nn, x, y):
    # Algoritmo de optimización
    optimizer = nn.GradientDescent(lr=0.001)

    # Optimización
    trainer = nn.SupervisedTrainer(model, optimizer, mean_cross_entropy_error, epochs=100, batch_size=8)
    history = trainer.train(x, y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Por último, podemos utilizar y evaluar el modelo:
    * El método `forward` permite obtener la salida de un modelo.
        * Para la clase Sequential, que está compuesta por varias capas, `forward` devuelve la salida de la última capa, sin el error
        * Para un problema de clasificación, debemos calcular el argmax ya que la salida son probabilidades de clase para cada ejemplo.

    Además, `edunn` tiene algunas funcionalidades extra para simplificar el uso de las redes:

    * El módulo `metrics` tiene algunas funciones para evaluar métricas de desempeño del mismo.
    * El módulo `plot` tiene algunas funciones para monitorear el entrenamiento del modelo (`plot_history`) y, en el caso en que el problema sea de pocas dimensiones (1 o 2), también visualizar las fronteras de decisión o la función ajustada (`plot_model_dataset_2d_classification`)
    """)
    return


@app.cell
def _(dataset_name, din, model, nn, x, y):
    # nn.plot.plot_history(history)

    # Reporte del desempeño
    y_pred = model.forward(x)
    y_pred_labels = y_pred.argmax(axis=1)
    print(f"Accuracy final del modelo: {nn.metrics.accuracy(y,y_pred_labels)*100:0.2f}%")


    if din ==2:
        # Visualización del modelo, solo si tiene 2 dimensiones
        nn.plot.plot_model_dataset_2d_classification(x,y,model,title=dataset_name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Como habrás notado, si bien pudimos definir la red y ejecutar el método `optimize` para pedirle al modelo que se entrene con descenso de gradiente, el accuracy obtenido es muy malo, es decir ¡la red no aprende!

    Esto _no_ es un error: no están implementados ninguno de los métodos correspondientes de los modelos (Bias, Linear, etc) ni el optimizador `GradientDescent`.

    **Tu** tarea será implementar las distintas capas/modelos de la librería `edunn`, así como algunos inicializadores y algoritmos de optimización, para que este código funcione.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementación de referencia

    El [repositorio de edunn](https://github.com/facundoq/edunn) contiene una implementación de referencia, que se enfoca en ser fácil de entender, y no en la eficiencia de cómputo.

    En base al código de esa implementación de referencia, y un programa que lo procesa, se generó una versión de edunn en donde se quitaron partes cruciales de la implementación de cada capa y otras clases.

    Para poder reimplementar la librería, tendrás que buscar las líneas de código entre los comentarios

    ```\"\"\" YOUR IMPLEMENTATION START \"\"\"```

    y

    ```\"\"\" YOUR IMPLEMENTATION END \"\"\"```

     y completar con el código correspondiente.

    En todos los casos, es importante enfocarse en buscar una implementación fácil de entender y que sea correcta, y dejar de lado la eficiencia para una implementación posterior.

    Si bien esta guía de implementación está en español, la implementación de la librería se ha realizado en inglés para que sea más fácil relacionar los conceptos con los de otras librerías.

    Los siguientes notebooks te guiarán en la implementación de cada `Model' (modelo), tanto en el método forward y el backward, y métodos importantes de otras clases.

    En caso de duda, siempre puedes consultar una solución posible en la [implementación de referencia](https://github.com/facundoq/edunn/tree/main/edunn).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plan de IMPLEMENTACION

    Comenzarás implementando dos *capas* muy simples: `AddConstant` y `MultiplyConstant`. La palabra *capa* es algo equivalente a *modelo* pero enfatiza que el modelo está destinado a combinarse con otra *capa* para formar un *modelo* más grande.

    En particular, estas dos capas realizan funciones muy simples y no tienen parámetros:

    * `AddConstant` agrega una constante a una matriz
    * `MultiplyConstant` multiplica una matriz por una constante.

    Por lo tanto, la implementación de los métodos `forward` y `backward` correspondientes será sencilla y te permitirá comenzar a familiarizarse con "edunn" y la metodología.

    Después de eso, comenzaremos a implementar capas más complejas, como `Bias` y `Linear` para formar modelos `LinearRegression`, con su función de error más común, `SquaredError`, y un `MeanError` para promediar el valor por muestra. errores en un lote de muestras.
    En ese punto, también implementaremos un optimizador `GradientDescent` para poner a prueba nuestros modelos. Después de eso, nos sumergiremos en capas más complejas, como "Softmax", para realizar la clasificación. Finalmente, implementaremos el modelo "Secuencial" para componer varias capas en una red neuronal modular completa.

    ¡Esperamos que te diviertas mucho programando tu primera red neuronal modular! 🕸️🚀
    """)
    return


if __name__ == "__main__":
    app.run()
