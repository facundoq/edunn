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
    # Cross Entropy Layer

    In this exercise, you need to implement the `CrossEntropyWithLabels` error layer.

    The cross-entropy function allows you to calculate the error of a model that outputs probabilities in terms of distances between distributions. In this case, we will measure the distance between the probability distribution that the model outputs, vs the true probability distribution.

    <img src="img/distance_prob.png" width="100%">

    In this case, `WithLabels` indicates that the true probability distribution (obtained from the dataset) is actually encoded with labels.

    For example, for a problem with `C=3` classes, if an example is of class 2 (counting from 0), then its label is `2`.

    This is a convenient way to specify that its encoding as a probability distribution would be `[0,0,1]`, which is a vector of `3` elements, where element `2` (again, counting from 0) has a probability of 1, and the rest are 0.

    Note that this is a special case of a probability distribution which is valid but  where there's actually not much of a "distribution". All the probability is concentrated on a single value, and for most practical purposes there's not much variability in the distribution. This is called a *degenerate* or *deterministic* distribution.

    This happens because our samples belong to only one class and we know exactly which one. The Cross Entropy loss actually allows measuring distances between arbitrary distributions, and does not require one of those to be *deterministic*. However, for now we'll implement the *WithLabels* version of the Cross Entropy that assumes that the true distribution is degenerate, and therefore makes the implementation a lot simpler.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Forward Method

    The `forward` method of the `CrossEntropyWithLabels` layer assumes that the input `y` is a probability distribution, i.e., `C` positive values that sum to 1, where `C` is the number of classes. Similarly, `y_true` is a label indicating which class out of the `C` classes is correct.

    For example, if $y=(0.3,0.4,0.3)$ and $y_{true}=2$, then there will be a considerable error since the value $y_{true}=2$ indicates that the distribution $y=(0,0,1)$ is expected. So, the values $0.3$ and $0.4$ for classes 0 and 1 should decrease, and the value $0.3$ for class 2 should increase.

    Cross entropy quantifies this error by calculating the negative logarithm of the probability of the correct class ($-ln(y_{y_{true}})$), in this case, class 2 ($-ln(y_2)$). So,

    $$CrossEntropy(y,y_{true}) = CrossEntropy((0.3,0.4,0.3),2) = -ln(0.3) = 1.20$$

    Again, in this case, the value $0.3$ was chosen because it is at index 2 of the vector $y$, meaning another way to write the above would be:

    $$E(y,y_{true}) = -ln(y_{y_{true}}) = -ln(0.3) = 1.20$$

    The reason for using the function $-ln(0.3)$ to penalize is that if the probability for the correct class is 1, then

    $$-ln(y_{y_{true}}) = -ln(1) = -0 = 0$$

    and there is no penalty. Otherwise, the output of $-ln$ will be positive and indicate an error. This way, it penalizes that the probability of the correct class does not reach 1. This can be visualized easily in a graph of the function $-ln(x)$:

    <img src="img/cross_entropy.png" width="400">

    Finally, since the values of $y$ are normalized, it is not necessary to penalize that the rest of the probabilities are greater than 0; if the error leads to the probability of the correct class being 1, then the rest must be 0. For this reason (among others), cross entropy is a good combination with the softmax function for training classification models.

    In the case of a batch of examples, the calculation is independent for each example.

    Implement the `forward` method of the `CrossEntropyWithLabels` class:
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
    # Backward Method

    Since the derivation of the equations for the `backward` method of cross entropy is a bit long, we provide [this note](http://facundoq.github.io/edunn/material/crossentropy_derivative) with the derivation of all cases.

    Again, since this error is for each example, the calculations are independent for each row.

    Implement the `backward` method of the `CrossEntropyWithLabels` class:
    """)
    return


@app.cell
def _(nn):
    # Number of random values of x and δEδy to generate and test gradients
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
    # Logistic Regression Applied to Flower Classification

    Now that we have all the elements, we can define and train our first logistic regression model to classify flowers in the [Iris dataset](https://www.kaggle.com/uciml/iris).

    Now, we can do it with Cross Entropy; although in this case, the results in terms of accuracy are similar, the model has a convex error, making optimization easier.
    """)
    return


@app.cell
def _(nn):
    # Load data with labels as outputs
    # (note: class labels start at 0)
    x, _y, classes = nn.datasets.load_classification('iris')
    # Normalize the data
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    n, din = x.shape
    # Calculate the number of classes
    classes = _y.max() + 1
    print('Sizes of x and y:', x.shape, _y.shape)
    model = nn.LogisticRegression(din, classes)
    # Logistic Regression model, 
    # with `din` input dimensions (4 for Iris)
    # and `classes` output dimensions (3 for Iris)
    error = nn.MeanError(nn.CrossEntropyWithLabels())
    # Mean Squared Error
    optimizer = nn.GradientDescent(lr=0.1)
    trainer = nn.SupervisedTrainer(model, optimizer, error, epochs=1000, batch_size=32)
    history = trainer.train(x, _y)
    # Optimization algorithm
    # nn.plot.plot_history(history, error_name=error.name)
    print('Model Metrics:')
    y_pred = model.forward(x)
    y_pred_labels = nn.utils.onehot2labels(y_pred)
    nn.metrics.classification_summary(_y, y_pred_labels)
    return


if __name__ == "__main__":
    app.run()
