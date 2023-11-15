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


class MSELoss:

    def __init__(self):
        self.label = None
        self.output = None

    def __call__(self, label, output, network=None):
        self.label = label
        self.output = output
        mse = 0.5*np.mean((label-output)**2, axis=1)
        return mse

    def gradient(self):
        grad_y = (self.output-self.label)/self.output.shape[1]
        return grad_y


class Sigmoid:
    # a = g(z)

    def __init__(self):
        self.z = None

    def __call__(self, z, network=None):
        self.z = z
        return 1/(1+np.exp(-z))

    def gradient(self, grad_a):
        sigma_z = 1 / (1 + np.exp(-self.z))
        grad_z = grad_a * sigma_z * (1 - sigma_z)
        return grad_z


class BCELoss:

    def __init__(self):
        self.label = None
        self. output = None

    def __call__(self, label, output, network=None):
        self.label = label
        self.output = output
        bce = -np.mean(label*np.log(output)+(1-label)*np.log(1-output), axis=1)
        return bce

    def gradient(self):
        grad_a = (self.output-self.label)/(self.output*(1-self.output))/self.output.shape[1]
        return grad_a


