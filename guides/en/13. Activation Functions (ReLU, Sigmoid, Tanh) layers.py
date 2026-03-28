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
    # Activation Functions

    A `neural network` is a model that combines transformations of various layers such as Linear Regression or Logistic Regression. However, to define a network, we must use _activation functions_, which allow for non-linear transformations of data. Otherwise, combining two layers of Linear Regression, for example, is equivalent to having only one layer since it can represent only a straight line. In this case, our network will not be more powerful, and it will only be more inefficient.

    There are different types of activation functions: we will see three of the most common ones, `relu`, `sigmoid` (or `logistic`), and `tanh` (hyperbolic tangent). Each one has different properties:

    * `relu`: It is efficient to calculate both its output and its derivative, and although its non-linearity is simple, it helps give the network greater approximation power. However, its output is in the range $[0,\infty)$, so it is not suitable for some tasks.
    * `sigmoid`: It is efficient, although not as much as `relu`, and since its range is $(0,1)$, it can encode a value that can be interpreted as a probability.
    * `tanh`: It is similar to `sigmoid`, but its range is $(-1,1)$, allowing it to encode values as positive and negative, generating representations with feedback of these signs in the network.

    <img src="img/relu.png" width="25%" style="display:inline-block"> <img src="img/sigmoid.png" width="25%" style="display:inline-block"> <img src="img/tanh.png" width="25%" style="display:inline-block">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ReLU Layer

    The `ReLU` (Rectified Linear Unit) function is extremely simple.

    <img src="img/relu.png" width=50%>

    Formally,

    $$ReLU(x) = \begin{cases}
        0 & \text{if } x \le 0 \\
        1 & \text{if}  x > 0
    \end{cases}
    $$

    In terms of code, calculating `relu` for a single value is simple:

    ```python
    def relu(x: float):
        if x > 0:
            return x
        else:
            return 0
    ```

    In a shorter but slightly more difficult-to-understand form, we can write `ReLU` as:
    $$ReLU(x) = \max(0,x)$$

    which also translates in code as

    ```python
    def relu(x:float):
        return max(x,0)
    ```

    Keep in mind that you will receive a tensor of values. Nevertheless, `ReLU`, like other activation functions, is easy to calculate because it is applied _element-wise_. That is, `ReLU` of a vector is equivalent to applying `ReLU` to each of its values, as follows:

    $$ReLU((-2,4,7)) = ( ReLU(-2), ReLU(4), ReLU(7)) = (0,4,7)$$

    For the case of a matrix or a tensor in general, the process is the same, and numpy operators make it easy to generalize this to arbitrary tensors.

    Implement the `forward` method of the `ReLU` class in the `edunn/models/activations.py` file.
    """)
    return


@app.cell
def _(nn, np):
    _x = np.array([[3.5, -7.2, 5.3], [-3.5, 7.2, -5.3]])
    _layer = nn.ReLU()
    _y = np.array([[3.5, 0, 5.3], [0, 7.2, 0]])
    nn.utils.check_same(_y, _layer.forward(_x))
    # plot values
    nn.plot.plot_activation_function(_layer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward Method of `ReLU`

    Fortunately, the `backward` method of `ReLU` is simple. By considering cases, we can see that if the forward method is:

    $$ReLU(x) = \begin{cases}
        0 & \text{if } x \le 0 \\
        x & \text{if } x > 0
    \end{cases}
    $$

    We can derive each case separately, and then:
    $$\frac{d ReLU(x)}{dx} = \begin{cases}
        0 & \text{if } x \le 0 \\
         1 & \text{if } x > 0
    \end{cases}
    $$

    In the case of $x=0$, the derivative of $ReLU$ is not actually defined; however, in this case, we can redefine that derivative as 0 (or 1), and it will not significantly affect optimization.
    """)
    return


@app.cell
def _(nn):
    from edunn.utils import check_gradient
    _samples = 100
    _input_shape = (5, 2)
    # Number of random examples and their size to generate
    # samples of x and check derivatives
    _layer = nn.ReLU()
    check_gradient.common_layer(_layer, _input_shape, samples=_samples)
    # Check derivatives of a ReLU function
    nn.plot.plot_activation_function(nn.ReLU(), backward=True)
    return (check_gradient,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sigmoid Layer

    The `sigmoid` function converts any value to the interval $(0,1)$.

    <img src="img/sigmoid.png" width=50%>

    Its definition is:

    $$
    Sigmoid(x)= \frac{1}{1+e^{-x}}
    $$

    For example:

    $$ sigmoid(0)=\frac{1}{1+1}=\frac{1}{2}=0.5$$
    $$ sigmoid(1)=\frac{1}{1+e^{-1}}=\frac{1}{1+0.36}=0.73$$
    $$ sigmoid(-1)=\frac{1}{1+e^{-(-1)}}=\frac{1}{1+2.71}=0.26$$

    As we can see, the balance of the function is at the value $0$, for which the output is $0.5$; values greater than $0$ result in greater outputs, and vice versa. As a curiosity,  $sigmoid(x)=1-sigmoid(-x)$.

    Implement the `forward` method of the `Sigmoid` class in the `edunn/models/activations.py` file.
    """)
    return


@app.cell
def _(nn, np):
    _x = np.array([[0, 1, -1]])
    _layer = nn.Sigmoid()
    _y = np.array([[0.5, 0.73105858, 0.26894142]])
    nn.utils.check_same(_y, _layer.forward(_x), tol=1e-06)
    nn.plot.plot_activation_function(_layer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward Method of `Sigmoid`

    The derivatives of the `backward` method of `Sigmoid` are simple, but we also want to simplify them further to make their computation more efficient.

    $$
    Sigmoid(x)= \frac{1}{1+e^{-x}}
    $$

    With a bit of algebra, we can see get to:
    $$
    \begin{aligned}
    \frac{d Sigmoid(x)}{dx}
    &= \frac{d \frac{1}{1+e^{-x}}}{dx} =\frac{d (1+e^{-x})^{-1}}{dx} \\
    &= \frac{d (1+e^{-x})^{-1}}{d(1+e^{-x})} \frac{d (1+e^{-x})}{dx} & \text{(chain rule with $g(x)=1+e^{-x}$)} \\
    &= \frac{d  (1+e^{-x})^{-1} }{d(1+e^{-x})} (-e^{-x}) & \text{(derivative of $1+e^{-x}$)} \\
    &=  -(1+e^{-x})^{-2} (-e^{-x}) & \text{(derivative of $g(x)^{-1}=-g(x)^{-2}$)} \\
    &=  (1+e^{-x})^{-2} e^{-x}\\
    \end{aligned}
    $$

    Wait, that's *almost* another sigmoid in there. Let's factor it out:

    $$
    \begin{aligned}
    (1+e^{-x})^{-2} e^{-x}
    &=  (\frac{1}{1+e^{-x}})² e^{-x}\\
    &=  Sigmoid(x)^2 e^{-x}\\
    \end{aligned}
    $$

    At this point, we have a nice formula for the derivative of `Sigmoid`, but as mentioned earlier. Plus, it can use the original value of `Sigmoid` for efficiency! However, to reduce numerical errors for small values of $Sigmoid(x)$ (when $x<<0$), we'll want to rewrite this a bit. Thus, we start with the last line of the above derivation:

    $$
    \begin{aligned}
    \frac{d Sigmoid(x)}{dx} &=  (1+e^{-x})^{-2} e^{-x}\\
    &=  Sigmoid(x)^2 e^{-x}\\
    &=  Sigmoid(x)^2 (1-1+e^{-x})\\
    &=  Sigmoid(x)^2 (1-Sigmoid(x)^{-1})\\
    &=  Sigmoid(x) Sigmoid(x) (1-Sigmoid(x)^{-1})\\
    &=  Sigmoid(x) [Sigmoid(x) (1-Sigmoid(x)^{-1})]\\
    &=  Sigmoid(x) [Sigmoid(x) * 1- Sigmoid(x) Sigmoid(x)^{-1}]\\
    &=  Sigmoid(x) (Sigmoid(x) -1)\\
    \end{aligned}
    $$

    This final formula $Sigmoid'(x) = Sigmoid(x)  (Sigmoid(x)-1)$ tells us that if we store the value of `Sigmoid(x)` during the `forward` pass, then for the `backward` pass, the derivative is simply calculated as $Sigmoid(x) (Sigmoid(x)-1)$, which only requires vector addition and multiplication.

    Implement the `backward` method of the `Sigmoid` class and verify it with the following code:
    """)
    return


@app.cell
def _(check_gradient, nn):
    _samples = 100
    _input_shape = (5, 2)
    _layer = nn.Sigmoid()
    check_gradient.common_layer(_layer, _input_shape, samples=_samples)
    nn.plot.plot_activation_function(_layer, backward=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # TanH Layer

    The `tanh` (hyperbolic tangent) function converts any value to the interval $(-1,1)$.

    <img src="img/tanh.png" width=35%>

    $tanh$ is a [hyperbolic](https://en.wikipedia.org/wiki/Hyperbolic_functions) geometric function. Like $tan(x)=\frac{cos(x)}{sin(x)}$, $tanh$ is defined in terms of hyperbolic cosine and sine:

    $$
    tanh(x)= \frac{cosh(x)}{sinh(x)} = \frac{e^x-e^{-x}}{e^x+e^{-x}}
    $$

    For example:

    $$
    \begin{aligned}[t]
    tanh(0)  &= \frac{1-1}{1+1} &&= \frac{0}{2} &&= 0 \\
    tanh(1)  &= \frac{e-e^{-1}}{e+e^{-1}} &&= \frac{2.35}{3.08} &&= 0.76 \\
    tanh(1) &= \frac{e^{-1}-e}{e^{-1}+e} &&= \frac{-2.35}{3.08} &&= -0.76
    \end{aligned}
     $$

    The balance of the function is at the value $0$, for which the output is $0$; values greater than $0$ result in greater outputs, and vice versa. Therefore, `tanh` is an odd function: $tanh(x)=-tanh(-x)$.

    $TanH$ is very similar to the `Sigmoid` function. In fact, with the following graph, we can see that $TanH$ is simply `Sigmoid`, but:

    * Multiplied by two (to convert the range $(0,1)$ to $(0,2)$).
    * Minus 1 (to convert the range $(0,2)$ to $(-1,1)$).
    * Multiplying $x$ by 2, so that we compress the $x$ axis and the curves are the same.

    <img src="img/sigmoid.png" width=35%>

    Therefore [it can be defined](http://facundoq.github.io/edunn/material/en/sigmoid_tanh) directly based on `Sigmoid`:

    $$
    tanh(x) = sigmoid(2x)*2-1
    $$

    This form will be much more convenient for implementation, as we can reuse the `Sigmoid` layer for both the forward and backward passes.

    Implement the `forward` method of the `TanH` class in the `edunn/models/activations.py` file _using the `Sigmoid` layer_ (we have already defined a variable for it).
    """)
    return


@app.cell
def _(nn, np):
    _x = np.array([[0, 0.5, -0.5]])
    _layer = nn.TanH()
    _y = np.array([[0.0, 0.46211716, -0.46211716]])
    nn.utils.check_same(_y, _layer.forward(_x), tol=1e-06)
    nn.plot.plot_activation_function(_layer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Backward Method of `TanH`

    The derivatives of the `backward` method of `TanH` can be obtained based on those of `Sigmoid`. Since

    $$
    tanh(x) = sigmoid(2x)*2-1
    $$

    Then:

    $$
    tanh'(x) = (sigmoid'(2x)*2)*2 = sigmoid'(2x)*4
    $$

    In other words, the derivative of `tanh` simply consists of multiplying the derivative of `Sigmoid` by two.

    Implement the `backward` method of the `TanH` class and verify it with the following code:
    """)
    return


@app.cell
def _(check_gradient, nn):
    _samples = 100
    _input_shape = (5, 2)
    _layer = nn.TanH()
    check_gradient.common_layer(_layer, _input_shape, samples=_samples)
    nn.plot.plot_activation_function(_layer, backward=True)
    return


if __name__ == "__main__":
    app.run()
