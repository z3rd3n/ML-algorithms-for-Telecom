import nn
import numpy as np
from matplotlib import pyplot as plt

layer1 = nn.Linear(1, 1)
criterion = nn.MSELoss()

x = np.random.randn(1, 1000)
y = 3 * x + 4 + np.sqrt(0.9) * np.random.randn(1, 1000)

num_epochs = 1000
loss_vec = []
w_vec = []
b_vec = []

plt.figure(1)
plt.plot(y[0], x[0], 'x', label='Pilots (Channel Input)')
bottom, top = plt.ylim()
init_output = layer1(np.sort(y))[0]
plt.plot(np.sort(y)[0], layer1(np.sort(y))[0], label='Initial Weights')
plt.xlabel('Channel Output')
plt.ylabel('Equalizer Output')
plt.legend()
plt.ylim(bottom, top)
plt.title('Equalizer Input v. Output (before training)')
plt.grid()
plt.show()

for epoch in range(num_epochs):
    # ---------------------------------------------- Task ----------------------------------------------
    # Do forward propagation through the layer and calculate the corresponding cost. Store the cost in a
    # variable called loss.

    # Your code goes here.
    output = layer1(x)
    loss = criterion(y, output)  # mse calculation
    # --------------------------------------------------------------------------------------------------

    loss = loss[0, 0] if len(loss.shape) > 1 else loss
    loss_vec.append(loss)
    w_vec.append(layer1.w[0, 0])
    b_vec.append(layer1.b[0, 0])

    # ---------------------------------------------- Task ----------------------------------------------
    # Reset the gradients of the layer. Calculate the gradient of the loss function as well as the gradients of the
    # linear layer. Update the weights afterward. Choose the step size beta by testing.

    # Your code goes here.
    layer1.reset_gradients()
    layer1.gradient(criterion.gradient())
    layer1.update_weights(beta=0.01)
    # --------------------------------------------------------------------------------------------------

# ---------------------------------------------- Task ----------------------------------------------
# Calculate optimal solutions mse_opt, w_opt and b_opt from Problem 1 a).

# Your code goes here.
w_opt = (np.mean(np.dot(x, y.T)) - np.mean(x)*np.mean(y))/(np.mean(y**2)-np.mean(y)**2)
b_opt = np.mean(x) - w_opt*np.mean(y)
mse_opt = np.mean(x**2) - np.mean(x)**2 - (np.mean(np.dot(x, y.T)-np.mean(x)*np.mean(y))**2)/(np.mean(y**2)-np.mean(y)**2)

# --------------------------------------------------------------------------------------------------

plt.figure(2)
plt.subplot(2, 2, 1)
plt.plot(y[0], x[0], 'x', label='Pilots (Channel Input)')
bottom, top = plt.ylim()
plt.plot(np.sort(y)[0], init_output, label='Initial Weights')
plt.plot(np.sort(y)[0], layer1(np.sort(y))[0], label='Trained Weights')
plt.xlabel('Channel Output')
plt.ylabel('Equalizer Output')
plt.legend()
plt.ylim(bottom, top)
plt.title('Equalizer Input v. Output (after training)')
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(loss_vec, label='Loss')
plt.plot([0, num_epochs - 1], [mse_opt, mse_opt], label='MMSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title('Comparison of Loss and LMMSE')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(w_vec, label='Weight')
plt.plot([0, num_epochs - 1], [w_opt, w_opt], label='Optimal Weight')
plt.xlabel('Epoch')
plt.ylabel('Weight')
plt.legend()
plt.title('Comparison of Weight and LMMSE Weight')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(b_vec, label='Bias')
plt.plot([0, num_epochs - 1], [b_opt, b_opt], label='Optimal Bias')
plt.xlabel('Epoch')
plt.ylabel('Bias')
plt.legend()
plt.title('Comparison of Bias and LMMSE Bias')
plt.grid()

plt.show()
