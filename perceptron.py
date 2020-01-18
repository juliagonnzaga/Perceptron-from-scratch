import random

class Perceptron():

    def __init__ (self, n_input, alpha=0.01):
        self.bias_weight = random.uniform(-1,1)
        self.alpha = alpha
        self.weights = []
        for i in range(n_input):
            self.weights.append(random.uniform(-1,1)) 

    def classify(self, input):
        summation = 0
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
        self.bias_weight += 1 * error * self.alpha
        for i in range(len(self.weights)):
            self.weights[i] += input[i] * error * self.alpha

perceptron = Perceptron(2)

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,0,0,1]

for iteration in range(100):
    print('Iteration ', iteration)
    for i in range(len(X)):
        guess = perceptron.classify(X[i])
        print(X[i], ' ', guess)
        perceptron.train(X[i],y[i])
    print('------------------')

print('0  0  |', perceptron.classify([0,0])) 
print('0  1  |', perceptron.classify([0,1])) 
print('1  0  |', perceptron.classify([1,0])) 
print('1  1  |', perceptron.classify([1,1])) 