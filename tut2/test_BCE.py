from nn import BCELoss
import numpy as np

loss = BCELoss()
x = np.array([[1., 1., 0., 0., 0., 0., 1., 0., 1., 1.]])
y = np.array([[0.67, 0.12, 0.43, 0.89, 0.75, 0.59, 0.13, 0.04, 0.2, 0.37]])
z_comp = np.array([1.23])


# Test cost function
z = np.round(loss(x, y), 2)

print("Testing class BCELoss.")

if z is None:
    raise Exception("Method __call__(self, label, output) does not provide a return value.")
if (not np.isscalar(z)) and (z.shape != (1,)) and (z.shape != (1, 1)):
    raise Exception("Method __call__(self, label, output) gives return value of wrong dimension. Expected dimensions "
                    + str((1,))+" or " + str((1, 1)) + " or " + str(()) + ", but got " + str(z.shape))
if np.linalg.norm(z-z_comp) > 0:
    raise Exception("Method __call__(self, label, output) does not model the behaviour correctly. Expected output is "
                    + str(z_comp.tolist()) + ", but got " + str(z.tolist()) + " (all rounded to 2 decimals).")
if (loss.label is None) | (np.linalg.norm(x-np.round(loss.label, 2)) > 0):
    raise Exception("Method __call__(self, label, output) didn't assign self.label correctly.")
if (loss.output is None) | (np.linalg.norm(y-np.round(loss.output, 2)) > 0):
    raise Exception("Method __call__(self, label, output) didn't assign self.output correctly.")


# Test gradient
g = np.round(loss.gradient(), 2)
g_comp = np.array([[-0.15, -0.83, 0.18, 0.91, 0.4, 0.24, -0.77, 0.1, -0.5, -0.27]])

if g is None:
    raise Exception("Method gradient(self) does not provide a return value.")
if not isinstance(g, np.ndarray):
    raise Exception("Method gradient(self) return value has wrong type. Should be of type np.ndarray.")
if g.shape != g_comp.shape:
    raise Exception("Method gradient(self) gives return value of wrong dimension. Expected dimensions "+str(g_comp.shape)
                    + ", but got " + str(g.shape))
if np.linalg.norm(g-g_comp) > 0:
    raise Exception("Method gradient(self) does not model the behaviour correctly. Expected output is "
                    + str(g_comp.tolist()) + ", but got " + str(g.tolist()) + " (all rounded to 2 decimals).")

print("All test passed succesfully.")
