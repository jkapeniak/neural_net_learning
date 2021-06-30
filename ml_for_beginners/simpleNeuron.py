import numpy as np

def sigmoid(x):
    # our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    
    def feedforward(self,inputs):
        #Weights inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

# weights = np.array([0,1]) #w1 = 0, w2 = 1
# bias = 0 # b =4
# n = Neuron(weights, bias)

# x = np.array([2,3]) # x1 = 2, x2 = 3
# print(n.feedforward(x))
# print(sigmoid(0.9526))


#Feedforward Neural Network


weights = np.array([0,1])
bias = 0
n = Neuron(weights,bias)
x = np.array([2,3])


h1 = n.feedforward(x)
h2 = n.feedforward(x)

o1Input = np.array([h1,h2])

o1 = n.feedforward(o1Input)

print(o1)