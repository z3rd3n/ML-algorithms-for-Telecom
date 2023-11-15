from nn import Linear
import numpy as np

# Don't Change!
in_size = 3
out_size = 2
n_sym = 2


layer1 = Linear(input_size=in_size, output_size=out_size)

print("Testing class Linear.")

# Check Weight Matrix
if layer1.w is None:
    raise Exception("Weight matrix not initialised.")
if not isinstance(layer1.w, np.ndarray):
    raise Exception("Weight matrix has wrong type. Should be of type np.ndarray.")
if layer1.w.shape != (out_size, in_size):
    raise Exception("Weight matrix has wrong dimensions. Expected dimensions "+str((out_size, in_size))+", but got "
                    + str(layer1.w.shape))

# Check Bias Vector
if layer1.b is None:
    raise Exception("Bias vector not initialised.")
if not isinstance(layer1.b, np.ndarray):
    raise Exception("Bias vector has wrong type. Should be of type np.ndarray.")
if layer1.b.shape != (out_size, 1):
    raise Exception("Bias vector has wrong dimensions. Expected dimensions "+str((out_size, 1))+", but got "
                    + str(layer1.b.shape))

# Check forward propagation
layer1.w = np.array([[0.2, 0.6, 0.1], [0.6, 0.3, 0.6]])
layer1.b = np.array([[0.4], [0.9]])
a = np.array([[-0.1, -0.4], [1.3, -2.7], [0.6, 0.1]])
z = np.round(layer1(a), 2)
z_comp = np.array([[1.22, -1.29], [1.59, -0.09]])

if z is None:
    raise Exception("Method __call__(self, a) does not provide a return value.")
if not isinstance(z, np.ndarray):
    raise Exception("Method __call__(self, a) return value has wrong type. Should be of type np.ndarray.")
if z.shape != (out_size, n_sym):
    raise Exception("Method __call__(self, a) gives return value of wrong dimension. Expected dimensions "
                    + str((out_size, n_sym)) + ", but got " + str(z.shape))
if np.linalg.norm(z-z_comp) > 0:
    raise Exception("Method __call__(self, a) does not model the behaviour correctly. Expected output is "
                    + str(z_comp.tolist()) + ", but got " + str(z.tolist()) + " (all rounded to 2 decimals).")
if (layer1.z is None) | (np.linalg.norm(z-np.round(layer1.z, 2)) > 0):
    raise Exception("Didn't assign self.z correctly during forward propagation")

if (layer1.a is None) | (np.linalg.norm(a-layer1.a) > 0):
    raise Exception("Didn't assign self.a correctly during forward propagation")


# Check gradient
grad_z = np.array([[1.0, -4.3], [3.3, -5.3]])
grad_a = np.round(layer1.gradient(grad_z), 2)
grad_a_comp = np.array([[2.18, -4.04], [1.59, -4.17], [2.08, -3.61]])
grad_w_comp = np.array([[1.62, 12.91, 0.17], [1.79, 18.6, 1.45]])
grad_b_comp = np.array([[-3.3, -2.]])

if grad_a is None:
    raise Exception("Method gradient(self, grad_z) does not provide a return value.")
if not isinstance(grad_a, np.ndarray):
    raise Exception("Method gradient(self, grad_z) return value has wrong type. Should be of type np.ndarray.")
if grad_a.shape != grad_a_comp.shape:
    raise Exception("Method gradient(self, grad_z) gives return value of wrong dimension. Expected dimensions "
                    + str(grad_a_comp.shape) + ", but got " + str(grad_a.shape)
                    + ". Check lecture notes for dimensions of gradients.")
if np.linalg.norm(grad_a-grad_a_comp) > 0:
    raise Exception("Method gradient(self, grad_z) does not model the behaviour correctly. Expected output is "
                    + str(grad_a_comp.tolist()) + ", but got " + str(grad_a.tolist()) + " (all rounded to 2 decimals).")

if layer1.grad_w is None:
    raise Exception("Method gradient(self, grad_z) does not assign self.grad_w.")
if not isinstance(layer1.grad_w, np.ndarray):
    raise Exception("Method gradient(self, grad_z) assigns self.grad_w wrong type. Should be of type np.ndarray.")
if layer1.grad_w.shape != grad_w_comp.shape:
    raise Exception("Method gradient(self, grad_z) assigns self.grad_w of wrong dimension. Expected dimensions "
                    + str(grad_w_comp.shape) + ", but got " + str(layer1.grad_w.shape)
                    + ". Check lecture notes for dimensions of gradients.")
if np.linalg.norm(np.round(layer1.grad_w, 2)-grad_w_comp) > 0:
    raise Exception("Method gradient(self, grad_z) assigns self.grad_w wrongly. Expected value is "
                    + str(grad_w_comp.tolist()) + ", but got " + str(np.round(layer1.grad_w, 2).tolist())
                    + " (all rounded to 2 decimals).")


if layer1.grad_b is None:
    raise Exception("Method gradient(self, grad_z) does not assign self.grad_b.")
if not isinstance(layer1.grad_b, np.ndarray):
    raise Exception("Method gradient(self, grad_z) assigns self.grad_b wrong type. Should be of type np.ndarray.")
if layer1.grad_b.shape != grad_b_comp.shape:
    raise Exception("Method gradient(self, grad_z) assigns self.grad_b of wrong dimension. Expected dimensions "
                    + str(grad_b_comp.shape) + ", but got " + str(layer1.grad_b.shape)
                    + ". Check lecture notes for dimensions of gradients.")
if np.linalg.norm(np.round(layer1.grad_b, 2)-grad_b_comp) > 0:
    raise Exception("Method gradient(self, grad_z) assigns self.grad_b wrongly. Expected value is "
                    + str(grad_b_comp.tolist()) + ", but got " + str(np.round(layer1.grad_b, 2).tolist())
                    + " (all rounded to 2 decimals).")


# Weight update
layer1.update_weights(0.1)
w_comp = np.array([[0.04, -0.69,  0.08], [0.42, -1.56,  0.45]])
b_comp = np.array([[0.73], [1.1]])


if np.linalg.norm(np.round(layer1.w, 2)-w_comp) > 0:
    raise Exception("Method update_weights(self, beta) assigns self.w wrongly. Expected value is "
                    + str(w_comp.tolist()) + ", but got " + str(np.round(layer1.w, 2).tolist())
                    + " (all rounded to 2 decimals).")
if np.linalg.norm(np.round(layer1.b, 2)-b_comp) > 0:
    raise Exception("Method update_weights(self, beta) assigns self.b wrongly. Expected value is "
                    + str(b_comp.tolist()) + ", but got " + str(np.round(layer1.b, 2).tolist())
                    + " (all rounded to 2 decimals).")

print("All test passed successfully.")
