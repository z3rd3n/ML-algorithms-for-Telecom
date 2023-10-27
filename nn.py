import numpy as np


# Some side notes:
# Here every vector defined like a matrix, for example
# Nx1 vector is of (N, ) but we define it like a matrix (N,1)
# to allow efficient matrix multiplication

class Linear:
    # z = Wa+b
    # it is run when layer1 = Linear(N, M)
    def __init__(self, input_size, output_size):
        self.w = np.random.rand(output_size, input_size)
        self.b = np.random.rand(output_size, 1)
        self.a = None
        self.z = None
        self.grad_w = None
        self.grad_b = None

    # it is run when layer1(a), gets a as input, and returns the z value
    def __call__(self, a):
        self.a = a
        self.z = self.w @ self.a + self.b
        return self.z

    def gradient(self, grad_z):
        self.grad_w = grad_z @ np.transpose(self.a)
        self.grad_b = np.transpose(grad_z @ np.ones((grad_z.shape[1], 1)))
        grad_a = np.transpose(self.w) @ grad_z
        return grad_a

    def update_weights(self, beta):
        self.w = self.w - beta * self.grad_w
        self.b = self.b - beta * np.transpose(self.grad_b)

    def reset_gradients(self):
        self.grad_w = None
        self.grad_b = None


class MSELoss:
    def __call__(self, label, output):
        self.label = None
        self.output = None

    def gradient(self):
        pass
