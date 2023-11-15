from nn import MSELoss
import numpy as np

loss = MSELoss()
x = np.array([[1.71, 0.38, -0.02, -0.69, -3.08, 0.62]])
y = np.array([[0.96, -0.29, -0.22, -0.44, 0.28, 0.61]])
z_comp = np.array([1.03])


# Test cost function
z = np.round(loss(x, y), 2)

print("Testing class MSELoss.")

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
g_comp = np.array([[-0.12, -0.11, -0.03,  0.04,  0.56, -0]])

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

print("All test passed successfully.")
