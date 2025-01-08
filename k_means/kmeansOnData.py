import numpy as np
import matplotlib.pyplot as plt
from kmeans_complex import k_means

np.random.seed(0)
y = np.load('EM_data.npy')
data= np.column_stack((y.real, y.imag))

# Apply K-means
M = 4  # You can change the number of clusters as needed
centroids, clusters = k_means(data.tolist(), M)

# Plotting
colors = plt.cm.viridis(np.linspace(0, 1, M))  # Using a colormap
for i, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    centroid_label = f'({centroids[i][0]:.2f}, {centroids[i][1]:.2f})'
    plt.scatter(cluster[:, 0], cluster[:, 1], s=30, c=[colors[i]], label=f'Cluster {centroid_label}')
plt.scatter(*zip(*centroids), s=100, c='black', marker='x')
plt.title('K-means Clustering')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.legend()
plt.show()
print(centroids)