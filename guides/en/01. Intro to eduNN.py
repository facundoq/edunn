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
    # **edunn Library**

    [edunn](https://github.com/facundoq/edunn) is a library for defining and training neural networks based on [Numpy](https://numpy.org/), designed to be simple to _understand_.

    Even more importantly, it was designed to be simple to _implement_. In other words, its **main** use is as a learning tool to understand how modern neural networks are implemented in frameworks like [Keras](https://keras.io/) or [Pytorch](https://pytorch.org/).

    However, it is also simple to _use_. For example, to define and train a three-layer neural network with different activation functions for classification, we can write code very similar to what you'd write with those frameworks:
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

    # Model definition
    layers = [nn.Linear(din,10),
              nn.Bias(10),
              nn.ReLU(),

              nn.Linear(10,n_classes),
              nn.Bias(n_classes),
              nn.Softmax()
              ]

    model = nn.Sequential(layers)
    print("Network Architecture:")
    print(model.summary())

    error = nn.MeanError(nn.CrossEntropyWithLabels())
    # Optimization algorithm
    optimizer = nn.GradientDescent(lr=0.1)

    # Optimization algorithm
    print("Training network with gradient descent:")
    history = optimizer.optimize(model,x,y,error)

    # Performance report
    y_pred = model.forward(x)
    y_pred_labels = y_pred.argmax(axis=1)
    print(f"Final model accuracy on training set: {nn.metrics.accuracy(y,y_pred_labels)*100:0.2f}%")

    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prior Knowledge
    To be able to implement the library, we assume that you have already learned the basic concepts of neural networks:

    * Layers
        * Linear Layers
        * Activation Functions
        * Layer Composition
        * Forward and Backward Methods
    * Backpropagation Algorithm
    * Gradient Descent
        * Gradient Calculation
        * Basic Gradient Optimization
    * Batch Computing/Training

    We also assume knowledge of Python and Numpy, as well as experience in handling tabular and image datasets.

    # Library Components

    We describe the basic components of the library used in the previous code to provide context for the exercises you'll solve.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Module **datasets**

    The `edunn.datasets` module allows you to easily load some test datasets. These datasets will be used to verify and experiment with the models.
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

    print(f"The {dataset_name} dataset has {n} examples, {din} features per example, and {n_classes} classes: {classes}.")
    return dataset_name, din, n_classes, nn, x, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To see what other classification or regression datasets are available in the `datasets` module of `edunn` (accessed as `nn.datasets`), you can run `nn.datasets.get_classification_names()` and `nn.datasets.get_regression_names()` and obtain a list of names.
    """)
    return


@app.cell
def _(nn):
    print("Available classification datasets:")
    print(nn.datasets.get_classification_names())
    print()

    print("Available regression datasets:")
    print(nn.datasets.get_regression_names())
    print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # edunn Classes and Modules

    To use `edunn`, we import the library and call it `nn` to make it easier to type.
    ```python
    import edunn as nn
    ```

    The library has a fundamental class, `Model`. This is the superclass of the models/layers we will implement, and it defines two abstract methods for its subclasses to implement:

    * `forward(x)`: computes the output `y` given an input `x` (or various inputs).
    * `backward(δEδy)`: computes the error gradient with respect to the input (`δEδx`), using the error gradient with respect to the output (`δEδy`). If the model/layer has parameters, it also calculates the gradient with respect to these parameters.
        * `backward` allows for a decoupled implementation of the backpropagation algorithm.
        * By using the `backward` of a model, we will implement of optimizers such as gradient descent.

    The library has several classes for different layers/models:

    * The classes `Linear` and `Bias` allow you to create layers with the functions $wx$ and $x+b$, respectively. In these cases, $w$ and $b$ are parameters to optimize. By combining these layers, you can create a traditional dense layer that calculates $wx+b$.
    * The classes `TanH`, `ReLU`, and `Softmax` allow you to create layers with activation functions of those names.
    * The `Sequential` class allows you to create sequential networks, where the output of each layer is the input of the next layer, and there is only one initial and one final layer.

    Each of these classes is a subclass of `Model`, and therefore allows for the two fundamental operations, `forward` and `backward`. Going forward, we will use the word 'layer' as a synonym for 'model', meaning a subclass of `Model`. This terminology, while somewhat inaccurate, is standard in the field of neural networks.
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
    print("Model Summary:")
    print(model.summary())
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Error Layers/Models

    Models need to measure their error. For this, `edunn` also has some error layers/models, which receive two inputs in their `forward`, the one calculated by the network and the expected one. We have two types of layers:

    * Those that allow us to calculate the error of the network _for each example separately_, like `CrossEntropyWithLabels`, or `SquaredError`
    * Those that allow us to combine the errors of each example to generate a scalar error.
        * The `MeanError` layer allows you to calculate the average error of another error layer, such as the `CrossEntropyWithLabels` and `SquaredError` layers mentioned.
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
    # Optimizers

    To train a model, we can use an `Optimizer` object, whose `optimize` method allows us to train a model to minimize that error on this dataset.
    For this training, an optimization algorithm must be specified. In this case, we use simple gradient descent with the `GradientDescent` class, a learning rate of `0.1`, `100` epochs, and a batch size of 8.
    """)
    return


@app.cell
def _(mean_cross_entropy_error, model, nn, x, y):
    # Optimization algorithm
    optimizer = nn.GradientDescent(lr=0.001)

    # Optimization
    trainer = nn.SupervisedTrainer(model, optimizer, mean_cross_entropy_error, epochs=100, batch_size=8)
    history = trainer.train(x, y)
    return (history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we can use and evaluate the model:
    * The `forward` method allows us to obtain the output of a model.
        * For the Sequential class, which is composed of several layers, `forward` returns the output of the last layer, without the error
        * For a classification problem, we must calculate the argmax since the output is class probabilities for each example.

    In addition, `edunn` has some extra features to simplify the use of neural networks:

    * The `metrics` module has some functions for evaluating performance metrics.
    * The `plot` module has some functions for monitoring model training (`plot_history`) and, in the case of low-dimensional problems (1 or 2 dimensions), also visualizing decision boundaries or the fitted function (`plot_model_dataset_2d_classification`)
    """)
    return


@app.cell
def _(dataset_name, din, history, model, nn, x, y):
    # nn.plot.plot_history(history)

    # Performance report
    y_pred = model.forward(x)
    y_pred_labels = y_pred.argmax(axis=1)
    print(f"Final model accuracy: {nn.metrics.accuracy(y,y_pred_labels)*100:0.2f}%")


    if din == 2:
        # Model visualization, only if it has 2 dimensions
        nn.plot.plot_model_dataset_2d_classification(x,y,model,title=dataset_name)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you may have noticed, although we were able to define the network and run the `optimize` method to instruct the model to train with gradient descent, the accuracy obtained is very poor, meaning the network does not learn!

    This is **not** an error: none of the corresponding methods of the models (Bias, Linear, etc.) or the `GradientDescent` optimizer are yet implemented.

    **Your** task is to implement the various layers/models of the `edunn` library, as well as some initializers and optimization algorithms, so that this code works. You'll therefore implement modern neural networks from scratch.

    Another pedagogical choice in EduNN is that it requires you to manually implement the `backward methoid
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reference Implementation

    The [edunn repository](https://github.com/facundoq/edunn) contains a reference implementation, which focuses on being easy to understand rather than computational efficiency.

    Based on the code from that reference implementation and a program that processes it, a version of edunn was generated in which crucial parts of the implementation of each layer and other classes were removed.

    To be able to reimplement the library, you will need to search for the lines of code between the `\"\"\" COMPLETION BEGIN \"\"\"` and `\"\"\" COMPLETION END \"\"\"` comments and complete them with the corresponding code.

    In all cases, it is important to focus on finding an implementation that is easy to understand and correct, and to set aside efficiency for a later implementation.

    The following notebooks will guide you in implementing each `Model` (model), both in the forward and backward methods, and important methods of other classes.

    If in doubt, you can always refer to a possible solution in the [reference implementation](https://github.com/facundoq/edunn/tree/main/edunn). The reference implementation is a complete, un-optimized solution to this exercises you can use to check your implementation. Do note that it has not been optimized for performance, since learning neural network fundamentals is the only goal of this project.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Implementation Plan

    You'll begin by implementing two very simple *layers*: `AddConstant` and `MultiplyConstant`.  The word *layer* is somewhat equivalent to *model* but emphasizes that the model is intended to be combined with other *layer* to form a larger *model*.

    In particular, these two layers perform very simple functions and have no parameters:

    * `AddConstant` adds a constant to an array
    * `MultiplyConstant`  multiplies an array by a constant.

    Therefore, the implementation of the corresponding `forward` and `backward` methods will be straightforward, and will allow you to begin familiarizing yourself with `edunn` and the methodology.

    After those, we'll start implementing more complex layers, such as `Bias` and `Linear` to form `LinearRegression` models, with its most common error function, `SquaredError`, and a `MeanError` to average the per-sample errors over a batch of samples.
    At that point we'll also implement a `GradientDescent` optimizer to put our models to test. After that, we'll dive into more complex layers, such as `Softmax`, to perform classification. Finally, we'll implement the `Sequential` model in order to compose several layers into a full-fledged modular neural network.

    We hope you have a lot of fun hacking out your first modular neural network! 🕸️🚀
    """)
    return


if __name__ == "__main__":
    app.run()
