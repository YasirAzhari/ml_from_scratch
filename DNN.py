import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def cross_entropy(predicted, labels):
    return -np.sum(labels * np.log(predicted)) / len(labels)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


class Layer:
    def __init__(self, input_dim, output_dim, activation="relu"):
        self.weights = np.random.random((output_dim, input_dim)) - 0.5
        self.biases = np.zeros(output_dim)
        self.a = np.zeros((output_dim, 1))
        self.z = np.zeros((output_dim, 1))

        activation_functions = {"relu": (relu, relu_derivative), "sigmoid": (sigmoid, sigmoid_derivative),
                                "softmax": (softmax, None)}
        self.activation, self.derivative = activation_functions[activation]

    def forward(self, inputs):
        self.z = inputs @ self.weights.T + self.biases
        self.a = self.activation(self.z)


class DNN:
    def __init__(self, architecture):
        self.layers = architecture
        self.n = self.layers[0].weights.shape[0]

    def forward(self, inputs):
        self.layers[0].forward(inputs)
        for i in range(1, len(self.layers)):
            self.layers[i].forward(self.layers[i-1].a)

    def backprop(self, inputs, labels, alpha):
        errors = self.layers[-1].a - labels

        dldz = (2/self.n * errors) * self.layers[-1].derivative(self.layers[-1].z)
        for i in range(len(self.layers)-1, -1, -1):
            if i > 0
                dldw = (1 / self.n) * self.layers[i - 1].a.T @ dldz
            else:
                dldw = (1 / self.n) * inputs.T @ dldz
            dldb = 1 / self.n * np.sum(dldz, axis=0)

            self.layers[i].weights -= alpha * dldw.T
            self.layers[i].biases -= alpha * dldb

            if i > 0:
                dldz = dldz @ self.layers[i].weights * self.layers[i - 1].derivative(self.layers[i - 1].z)

    def train(self, inputs, labels, alpha, epochs):
        for epoch in range(epochs):
            self.forward(inputs)
            error = self.layers[-1].a - labels
            print((error ** 2).sum() / self.n)
            self.backprop(inputs, labels, alpha)
        print(self.layers[-1].a)


test_inputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
test_labels = np.array([[0], [0], [1], [1]])

architecture = [Layer(2, 2), Layer(2, 4), Layer(4, 1, activation="sigmoid")]
myMLP = DNN(architecture)
myMLP.train(test_inputs, test_labels, 0.3, 10000)