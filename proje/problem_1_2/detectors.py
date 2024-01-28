import numpy as np
import nn


def normpdf(x, mu, var):
    return 1/np.sqrt(2*np.pi*var)*np.exp(-1/2*(x-mu)**2/var)


def mu_func(mu, var, xi, yi, var_n):
    return (mu * var_n + xi * yi * var) / (var_n + var)

def var_func(var, var_n):
    return (var_n * var) / (var_n + var) + var_n + var

def detector_spa(const, y, var_h, mu_h, var_n):
    k = y.size
    norm_coeff = 1 / (2 ** k)
    multi = 1

    mu_init = mu_h
    var_init = var_h

    for i in range(k):
        xi, yi = const[i, :], y[i]
        multi *= normpdf(mu_init, xi * yi, var_init)
        var_init = var_func(var_init, var_n)
        mu_init = mu_func(mu_init, var_init, xi, yi, var_n)

    p_x_y = multi * norm_coeff 
    p_y = sum(p_x_y)

    return p_x_y[None, :] / p_y


class DetectorNN(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DetectorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x, self)
        x = self.sigmoid1(x, self)
        x = self.fc2(x, self)
        x = self.sigmoid2(x, self)
        x = self.fc3(x, self)
        x = self.softmax(x, self)
        return x

