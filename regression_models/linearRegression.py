import nn
import numpy as np

layer1 = nn.Linear(1, 1)
criterion = nn.MSELoss()

x = np.random.randn(1, 1000)
y = 3 * x + 4 + np.sqrt(0.9) * np.random.randn(1, 1000)

num_epochs = 1000

for epoch in range(num_epochs):
    output = layer1(y)  # here y is input to equalizer, thus input to network

    loss = criterion(x, output)  # mse calculation with output and our target

    layer1.gradient(criterion.gradient())

    layer1.update_weights(beta=0.03)

    layer1.reset_gradients()
