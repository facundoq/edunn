


### :brain: `simplenn` Learn to implement modern, modular Neural Networks by implementing a PyTorch/Keras-like framework.

`simplenn` is a very simple library that contains the basic elements to define, train and use **modular** Neural Networks using just Python and Numpy. 

We strip the key parts of `simplenn` that deal with layers, initializers, error functions, backpropagation and optimizers in the process. 

Through our guides, we help **you reimplement `simplenn`** almost *from scratch* with various tests along the way to ensure you are doing it correctly. 


Supported languages:

* [Spanish](releases/es)
* English (coming soon)


# :rainbow: Why `simplenn`?

Well-known courses that teach you how to program neural networks from the ground up such as [*CS231n: Convolutional Neural Networks for Visual Recognition*](https://cs231n.github.io/) or [Andrew Ng's *Machine Learning*](https://www.coursera.org/learn/machine-learning) are great for understanding the basics using `numpy`, but use monolithic neural network models that make it difficult to understand how that code translates to other frameworks such as `pytorch` or `tensorflow`.

Alternatively, [Andrew Ng's Deep Learning](https://www.coursera.org/specializations/deep-learning) or [FastAI](https://course19.fast.ai/part2) build networks out of `tensorflow` or basic `pytorch` operators instead of numpy. This is great to build complex models, but there's still a lot of magic under the hood since both frameworks automatically derive `backward` methods. 

 Using `simplenn` to understand how modern neural networks frameworks work fills the gap in between; you get full low-level control but without all the complexities `pytorch` or `tensorflow` bring along.


# :heavy_plus_sign: Pre-requisites

The guides are intended for learners with some experience with Python, Numpy and Neural Networks. We have included explanations of what you should implement, but learners still should be able to:

1. Read and write object-oriented python code, including subclassing and method overriding.
2. Understand Numpy, basic linear algebra and calculus and be able to translate  mathematical operations to Numpy code.
3. Understand the basic ideas in modern modular Neural Network frameworks, such as models, layers, optimizers, batches, backward and forward passes, and specially backpropagation. 

# :wrench: Download and install

**Do not clone** this repository if you want to solve the exercises. 

1. To use `simplenn` as a learning exercise, please download a [release](releases) for your language. If you like the command line, use `wget`, replacing `es` with your preferred language (only `es` currently supported):

    `wget https://github.com/facundoq/simplenn/releases/es.zip`

2. Unzip to `simplenn` or other folder name you choose:

    `unzip es.zip -d simplenn && cd simplenn`

3. Make sure Python3.6 or greater and pip3 are installed:

    `python3 --version`
    `pip3 --version`

4. Make sure virtualenv is installed or else install with virtualenv:

    `pip3 --user install virtualenv`

5. Create a virtualenv environment and install the dependencies in requirements.txt: 

    `python3 -m venv nnenv`
    `source nnvenv/bin/activate`
    `pip install -r requirements.txt`

# :pill: Solutions and bugs :bug: 

`simplenn` is designed so that you can automatically test your implementations. However, it is intended to be used as a set of exercises in a course or seminar. Please address all questions regarding the material to your instructor. Alternatively, you may consult public forums such as [stack overflow](stackoverflow.com/), [r/neuralnetworks](https://www.reddit.com/r/neuralnetworks/) or [r/MachineLearning](https://www.reddit.com/r/MachineLearning)

Please, only [file an issue](issues) if there is an actual bug or feature request.  

# :package: Reference implementation and usage as library

This repository  also has a reference implementation in the `simplenn` of the library you can consult.

You can `pip install simplenn` to use the reference implementation of the library to train models and use them. However, we **do not recommend** doing so since the library is *slow* and not meant for neither research nor production environments.  

# :busts_in_silhouette: Contributing

We'd like to expand `simplenn` with more guides/layers/optimizers. Check out our [contribution guide](CONTRIBUTE.md)! 
