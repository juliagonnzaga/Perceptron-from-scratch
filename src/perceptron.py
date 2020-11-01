import matplotlib.pyplot as plt
import numpy as np
import random

IMG_FOLDER = "../imgs/"


class Perceptron():

    def __init__(self, n_input, alpha=0.01, has_bias=True):
        self.has_bias = has_bias
        self.bias_weight = random.uniform(-1, 1)
        self.alpha = alpha
        self.weights = []
        for i in range(n_input):
            self.weights.append(random.uniform(-1, 1))

    def classify(self, input):
        summation = 0
        if(self.has_bias):
            summation += self.bias_weight * 1
        for i in range(len(self.weights)):
            summation += self.weights[i] * input[i]
        return self.activation(summation)

    def activation(self, value):
        if(value < 0):
            return 0
        else:
            return 1

    def train(self, input, target):
        guess = self.classify(input)
        error = target - guess
        if(self.has_bias):
            self.bias_weight += 1 * error * self.alpha
        for i in range(len(self.weights)):
            self.weights[i] += input[i] * error * self.alpha

# Plot y data for the plot_perceptron_func


def plot_y(fig, y):
    for i in range(len(y)):
        dot = 'ko' if y[i] > 0 else 'ro'
        plt.plot(X[i][0], X[i][1], dot)
    return fig

# Plot to illustrate the training


def plot_perceptron_func(p, y):
    h = .02
    x_min = -0.8
    x_max = 1.2
    y_min = -0.8
    y_max = 1.2
    xx = np.arange(x_min, x_max, h)
    yy = np.arange(y_min, y_max, h)
    fig, ax = plt.subplots()
    fig = plot_y(fig, y)
    Z = []
    for x in xx:
        z = []
        for y in yy:
            z.append(p.classify([x, y]))
        Z.append(z)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.axis([-0.2, 1.2, -0.2, 1.2])
    plt.savefig(IMG_FOLDER+str(iteration))

# Plot just the y data


def plot_dots(X, y):
    for i in range(len(y)):
        dot = 'ko' if y[i] > 0 else 'ro'
        plt.plot(X[i][0], X[i][1], dot)
    plt.axis([-0.2, 1.2, -0.2, 1.2])
    plt.show()

# Linear function


def func_linear(x1, a, b):
    return a * x1 + b

# Sigmoid function


def sigmoid(x):
    return 1 / (1 + e ** -x)

# Plot activation function


def plot_activation_function(a=1, b=1, color='', sig=False):
    X = np.arange(-10, 10, 0.05)
    Y = sigmoid(func_linear(X, a, b)) if sigmoid else func_linear(X, a, b)
    label = 'f(x)='+str(a)+'x'+'+'+str(b)
    plt.plot(X, Y, '--', color=color, label=label)


##
# Setting Paerceptron and data
##
perceptron = Perceptron(2, has_bias=True)
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 1]


##
# Training
##
for iteration in range(101):
    for i in range(len(X)):
        perceptron.train(X[i], y[i])
    if(iteration % 5 == 0):
        plot_perceptron_func(perceptron, y)

##
# Testing
##
for i in range(len(X)):
    print(X[0], perceptron.classify(X[0]))
