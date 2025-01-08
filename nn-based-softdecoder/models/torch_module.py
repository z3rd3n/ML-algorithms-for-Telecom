import numpy as np


class Linear:
    # z = Wa+b

    def __init__(self, input_size, output_size):
        self.w = 2*np.sqrt(input_size)*np.random.rand(output_size, input_size)-np.sqrt(input_size)
        self.b = 2*np.sqrt(input_size)*np.random.rand(output_size, 1)-np.sqrt(input_size)
        self.a = None
        self.z = None
        self.grad_w = None
        self.grad_b = None

    def __call__(self, a, network=None):
        self.a = a
        self.z = self.b + self.w @ a
        if network is not None:
            if self not in network.layers:
                network.add_layer(self)
        return self.z

    def gradient(self, grad_z):
        self.grad_w = grad_z @ np.transpose(self.a)
        self.grad_b = np.transpose(grad_z @ np.ones((grad_z.shape[1], 1)))
        grad_a = np.transpose(self.w) @ grad_z
        return grad_a

    def update_weights(self, beta, lbd=0):
        self.w = self.w-beta*(self.grad_w+2*lbd*self.w)
        self.b = self.b-beta*(np.transpose(self.grad_b)+2*lbd*self.b)

    def reset_gradients(self):
        self.grad_b = None
        self.grad_w = None


class Sigmoid:
    # a = g(z)

    def __init__(self):
        self.z = None

    def __call__(self, z, network=None):
        self.z = z
        if network is not None:
            if self not in network.layers:
                network.add_layer(self)
        return 1/(1+np.exp(-z))

    def gradient(self, grad_a):
        sigma_z = 1 / (1 + np.exp(-self.z))
        grad_z = grad_a * sigma_z * (1 - sigma_z)
        return grad_z


class Module:

    def __init__(self):
        self.layers = []
        self.cost_function = None

    # Calculate and return outputs for given inputs.
    def __call__(self, inputs):
        return self.forward(inputs)

    # This is just a dummy method. Don't change it.
    def forward(self, inputs):
        pass

    #   Add a layer to the network's list of layers
    def add_layer(self, layer):
        self.layers.append(layer)

    # change the network's cost function
    def change_cost_function(self, cost_function):
        self.cost_function = cost_function

    # Reset gradients of all linear layers
    def zero_grad(self):
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.reset_gradients()

    # Perform backpropagation
    def backward(self):
        # Backpropagation logic
        grad_a = self.cost_function.gradient()

        for layer in reversed(self.layers):
            grad_a = layer.gradient(grad_a)

    # Update weights and biases in all linear layers
    def step(self, beta=0.1, lbd=0):
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                layer.update_weights(beta, lbd)


class CELossSoftmax:

    def __init__(self):
        self.label = None
        self. output = None

    def __call__(self, label, output, network=None):
        if network is not None:
            if network.cost_function is not self:
                network.change_cost_function(self)

        self.label = label
        self.output = output
        return -np.sum(label * np.log(output))/output.shape[1]
        

    def gradient(self):
        return (self.output-self.label)/self.output.shape[1]


class Softmax:
    # a = softmax(z)

    def __init__(self):
        self.z = None

    def __call__(self, z, network=None):
        self.z = z
        exp_z = np.exp(z)

        if network is not None:
            if self not in network.layers:
                network.add_layer(self)

        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def gradient(self, value_to_pass):
        return value_to_pass







