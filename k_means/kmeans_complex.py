import numpy as np
import matplotlib.pyplot as plt

# K-means functions as defined previously
def euclidean_distance(point1, point2):
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

def assign_points_to_clusters(centroids, points):
    clusters = [[] for _ in centroids]
    for point in points:
        shortest_distance = float('inf')
        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(point, centroid)
            if distance < shortest_distance:
                shortest_distance = distance
                closest_centroid = i
        clusters[closest_centroid].append(point)
    return clusters

def calculate_new_centroids(clusters):
    centroids = []
    for cluster in clusters:
        if cluster:
            centroids.append(tuple(sum(dim) / len(dim) for dim in zip(*cluster)))
        else:
            centroids.append(None)
    return centroids

def k_means(points, k, max_iterations=100):
    centroids = points[:k]
    for _ in range(max_iterations):
        clusters = assign_points_to_clusters(centroids, points)
        new_centroids = calculate_new_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids
    return centroids, clusters




