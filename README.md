


#  `edunn`

## :brain: Learn how modern, modular Neural Networks work by implementing a PyTorch/Keras-like framework.

`edunn` is a very simple library that contains the basic elements to define, train and use **modular** Neural Networks using just Python and Numpy ([full example](https://colab.research.google.com/github/facundoq/edunn/examples/iris_classification.ipynb)).

````python
import edunn as nn
x,y = load_data()
din,n_classes=4,10
layers = [nn.Linear(din,10),
          nn.Bias(10),
          nn.ReLU(),
          nn.Linear(10,n_classes),
          nn.Bias(n_classes),
          nn.Softmax()
          ]

model = nn.Sequential(layers)
print(model.summary())
error = nn.MeanError(nn.CrossEntropyWithLabels())
optimizer = nn.GradientDescent(lr=0.1,epochs=3000,batch_size=32)
history = optimizer.optimize(model,x,y,error)
nn.plot.plot_history(history)
````

We _remove_ the key parts of `edunn` that deal with layers, initializers, error functions, backpropagation and optimizers. Then, through our guides, we help **_you_ reimplement** the key parts of `edunn`. We also provide various automated tests along the way for you to check your solutions. 


Supported languages:

* [Spanish](https://github.com/facundoq/edunn/blob/main/releases/edunn-es.zip)
* English (coming soon)


## :rainbow: Why `edunn`?

There are many excellent courses and books that teach you how to program neural networks from the ground up such as [*CS231n: Convolutional Neural Networks for Visual Recognition*](https://cs231n.github.io/), Andrew Ng's [*Machine Learning*](https://www.coursera.org/learn/machine-learning) or Sebastian Rascha's [Python Machine Learning](https://sebastianraschka.com/books/#python-machine-learning-3rd-edition). However, while these are great for understanding the basics using `numpy`, they use monolithic neural network models that make it difficult to understand how that code translates to other frameworks such as `pytorch` or `tensorflow`.

Alternatively, [Andrew Ng's *Deep Learning*](https://www.coursera.org/specializations/deep-learning) or [*FastAI*](https://course19.fast.ai/part2) build modular networks out of `tensorflow` or basic `pytorch` operators instead of numpy. This is great for building complex models quickly, but there's still a lot of magic under the hood. Both frameworks automatically derive `backward` methods and bring along other goodies. 

Implementing `edunn` allows you to understand how modern neural networks frameworks work and fills the gap between those types of learning. You get to program a full neural network framework, without all the magic `pytorch` or `tensorflow` bring along. Of course, `edunn` guides help you out along the way and provide a clear path to complete the implementation of the library. Also, in this case you'd actually be _reimplementing_ `edunn`, since we provide a reference implementation you can check when in doubt with your solutions.


## :heavy_plus_sign: Pre-requisite knowledge

The guides are intended for learners with some experience with Python, Numpy and Neural Networks. We have included explanations of what you must implement, but learners still should be able to:

1. Read and write object-oriented python code, including subclassing and method overriding.
2. Understand Numpy, basic linear algebra and calculus and be able to translate  mathematical operations to Numpy code.
3. Understand the basic ideas in modern modular Neural Network frameworks, such as models, layers, optimizers, batches, backward and forward passes, and specially backpropagation. It is also helpful but not necessary to have previous exposure to `tensorflow/keras` or `pytorch`.

## :wrench: Download and setup

**Do not clone** this repository if you want to use `edunn` as a learning exercise. Instead, follow these steps to get a working environment for the guides. These instructions assume a working installation of python3 (3.6 or greater), pip3 and virtualenv. The installations of those vary wildly with OS/distribution so it's up to you to get a setup working.

Afterward: 

1. Download a [release](https://github.com/facundoq/edunn/blob/main/releases) for your language. If you prefer the command line, use `wget`, replacing `es` with your preferred language (only `es` is currently supported):

    `wget https://github.com/facundoq/edunn/blob/main/releases/edunn-es.zip`

2. Unzip to `edunn` or other folder name you choose:

    `unzip es.zip -d edunn && cd edunn`

3. Make sure Python3.6 or greater, pip3 and venv are installed:

    ````bash
    python3 --version
    pip3 --version && 
    python3 -c "import venv" && echo "venv is installed"
    ````
   

4. Create a virtualenv environment and install the dependencies in the requirements file `requirements.txt`: 

    ````bash
    python3 -m venv nnenv
    source nnvenv/bin/activate
    pip install -r requirements.txt
    ````

Alternatively, you can use your `conda` distribution or another such tool to create a virtual environment and install the required libraries for `edunn` (listed inside `requirements.txt`). 

5. Run jupyter and follow the guides:

```bash
jupyter notebook
```

## :pill: Solutions and bugs :bug: 

`edunn` is designed so that you can automatically test your implementations. However, it is intended to be used as a set of exercises in a course or seminar. Please address all questions regarding the material to your instructor.

This repository also has a [reference implementation](https://github.com/facundoq/edunn/tree/main/edunn) in the `edunn` folder of the library you can consult. 

Alternatively, you may consult public forums such as [stack overflow](stackoverflow.com/), [r/neuralnetworks](https://www.reddit.com/r/neuralnetworks/) or [r/MachineLearning](https://www.reddit.com/r/MachineLearning)

Please, **only [file an issue](issues) if there is an actual bug** or feature request, not if you can't solve the exercises.  

## :package: Reference implementation and usage as library


You can `pip install edunn` to use the reference implementation of the library to train models and use them. Indeed, the API is very simple:

````python
import edunn as nn

x, y, n_classes = ...  # load data 
n, din = x.shape

# Model definition
layers = [nn.Linear(din, 10),
          nn.Bias(10),
          nn.ReLU(),
          nn.Linear(10, 20),
          nn.Bias(20),
          nn.TanH(),
          nn.Linear(10, n_classes),
          nn.Bias(n_classes),
          nn.Softmax()
          ]

model = nn.Sequential(layers)
print(model.summary())

error = nn.MeanError(nn.CrossEntropyWithLabels())
optimizer = nn.StochasticGradientDescent(lr=0.001, epochs=1000, batch_size=32)
history = optimizer.optimize(model, x, y, error)
````

However, we **do not recommend** for serious projects since the library is *very slow* and not meant for neither research nor production environments. The reference implementation is mostly focused on being easy to understand so that learners can check their implementations.

## :busts_in_silhouette: Contributing

We'd like your help to expand `edunn` with more guides/layers/optimizers. Check out our [contribution guide](CONTRIBUTE.md)! 
