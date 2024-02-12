import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Load dataset
data = np.load('EM_data.npy')

# Scatter plot for initial insights
plt.scatter(data.real, data.imag, marker='.', alpha=0.5)
plt.title('Transmit vs. Receive Data')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.show()

def k_means_initialization(data, k):
    # Randomly initialize cluster centers
    cluster_centers = data[np.random.choice(len(data), k, replace=False)]
    
    for _ in range(20):  # Perform 20 iterations for simplicity
        # Assign each point to the nearest cluster
        distances = np.linalg.norm(data[:, None] - cluster_centers, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update cluster centers
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.linalg.norm(new_centers - cluster_centers) < 1e-4:
            break
        
        cluster_centers = new_centers
    
    return new_centers.flatten()  # Ensure the mean vector is flattened to a 1D array

# Initial parameters for EM
initial_params = {
    'sigma_squared': 1.0,
    'delta': complex(1.0, 1.0),
    'PX_mean': k_means_initialization(data, k=3),  # Adjust k value accordingly
    'PX_covariance': 1.0
}

# EM Algorithm
def expectation_maximization(data, initial_params, max_iter=100, tol=1e-6):
    # Unpack initial parameters
    sigma_squared = initial_params['sigma_squared']
    delta = initial_params['delta']
    PX_mean = initial_params['PX_mean']
    PX_covariance = initial_params['PX_covariance']

    for iteration in range(max_iter):
        # Expectation step
        responsibilities = multivariate_normal.pdf(data, mean=PX_mean, cov=PX_covariance).real
        responsibilities /= responsibilities.sum(axis=0, keepdims=True)

        # Maximization step
        # Maximization step
        N_k = responsibilities.sum(axis=0)
        PX_mean = (data.T @ responsibilities) / N_k

        # Ensure covariance matrix is positive semidefinite
        cov_update = (data - PX_mean).T @ (responsibilities[:, None] * (data - PX_mean)) / N_k
        eigenvalues, eigenvectors = np.linalg.eigh(cov_update)
        eigenvalues[eigenvalues < 0] = 0  # Set negative eigenvalues to zero
        PX_covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T




        # Check for convergence
        new_sigma_squared = np.mean(np.abs(data - delta * PX_mean) ** 2)  # Example formula, replace with your calculation
        new_delta = np.mean(data / PX_mean)  # Example formula, replace with your calculation

        if np.abs(sigma_squared - new_sigma_squared) < tol and np.abs(delta - new_delta) < tol:
            break

        # Update parameters
        sigma_squared = new_sigma_squared
        delta = new_delta

    # Return the estimated parameters
    return {
        'sigma_squared': sigma_squared,
        'delta': delta,
        'PX_mean': PX_mean,
        'PX_covariance': PX_covariance
    }



# Initial parameters for EM
initial_params = {
    'sigma_squared': 1.0,
    'delta': complex(1.0, 1.0),
    'PX_mean': k_means_initialization(data, k=3),  # Adjust k value accordingly
    'PX_covariance': 1.0
}

# Run EM
result_params = expectation_maximization(data, initial_params)

# Print results
print(f"Estimated Variance (sigma^2): {result_params['sigma_squared']}")
print(f"Estimated Delta (∆): {result_params['delta']}")
print(f"Estimated Distribution on Constellation (PX): Mean={result_params['PX_mean']}, Covariance={result_params['PX_covariance']}")
