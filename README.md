

# SimpleNN

### :brain: Learn to implement modern, modular Neural Networks by implementing a PyTorch/Keras-like framework.

`simplenn` is a very simple library that contains the basic elements to define, train and use **modular** Neural Networks. 

We strip the key parts of the code to let you reimplement them and learn about layers, initializers, error functions, backpropagation and optimizers.




Supported languages:

* [Spanish](releases/es)

[comment]: <> (* [English]&#40;https://www.countryflags.io/gb/shiny/32.png&#41;)

# :sparkles: Why SimpleNN?

Well-known courses that teach you how to program neural networks from the ground up such as [*CS231n: Convolutional Neural Networks for Visual Recognition*](https://cs231n.github.io/) or [Andrew Ng's *Machine Learning*](https://www.coursera.org/learn/machine-learning) are great for understanding the basics, but use monolithic neural network models that don't allow for modularity.

Alternatively, [Andrew Ng's Deep Learning](https://www.coursera.org/specializations/deep-learning) or [FastAI](https://course19.fast.ai/part2) build network out of tensorflow or basic pytorch operators, which  automatically derive `backward` methods. 



# :wrench: Download and install

**Do not** clone this repository if you want to solve the exercises. 

To use SimpleNN, please download a [release](releases) for your language or use wget, replacing `es` with your preferred language:

`wget https://github.com/facundoq/simplenn/releases/es.zip`

Unzip to `simplenn` or other foldername you choose:

`unzip es.zip -d simplenn`

Make sure Python3.6+ and pip:

`python3 --version`
`pip3 --version`

Install virtualenv:

`pip3 --user install virtualenv`

Create a Python3.6+ virtual env and install the dependencies in requirements.txt 

`python3 -m venv nnenv`
`source nnvenv/bin/activate`
`pip install -r requirements.txt`

# :pill: Solutions

SimpleNN is intended to be used as an exercise or supplementary material in a course or seminar. Please address all questions regarding the material to your instructor. Alternatively, you may consult public forums such as [stack overflow](stackoverflow.com/), [r/neuralnetworks](https://www.reddit.com/r/neuralnetworks/) or [r/MachineLearning](https://www.reddit.com/r/MachineLearning) 

This repository  also has a reference implementation in the `simplenn` of the library you can consult. 

# :: Reference implementation and usage as library

You can `pip install simplenn` to use the reference implementation of the library to train models and use them. However, we **do not recommend** doing so since the library is *slow* and not meant for neither research nor production environments.  

# Contributing

We are open to [contributions](CONTRIBUTE.md)! 
