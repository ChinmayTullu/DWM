import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('ENTIRE.csv', nrows=50)
X = data[['max_power', 'engine']].values
print("the data:\n", X)

# Set parameters
k = 3
iters = 100
np.random.seed(0)  # For reproducibility

# Initialize centroids randomly
indices = np.random.choice(len(X), k, replace=False)
centroids = X[indices]
print("\nthe centroids:\n", centroids)

# Perform K-means clustering
for _ in range(iters):
    # Step 1: Assign each point to the nearest centroid
    clusters = []
    for point in X:
        euclid_distances = [np.sqrt(np.sum((point - centroid) ** 2)) for centroid in centroids]
        closest_centroid = np.argmin(euclid_distances)
        clusters.append(closest_centroid)
    
    # Step 2: Update centroids
    new_centroids = []
    for i in range(k):
        # Get all points assigned to the current cluster
        cluster_points = [X[j] for j in range(len(X)) if clusters[j] == i]
        if cluster_points:
            # Calculate the mean of the points to find the new centroid
            new_centroids.append(np.mean(cluster_points, axis=0))
        else:
            # If a cluster has no points assigned, keep the previous centroid
            new_centroids.append(centroids[i])
    centroids = new_centroids
    
print("\nthe clusters:\n", clusters)

# Convert clusters and centroids to numpy arrays for plotting
clusters = np.array(clusters)
centroids = np.array(centroids)

# Plotting the clusters and centroids
plt.figure(figsize=(8, 6))

# Plot each cluster with a different color
for i in range(k):
    cluster_points = X[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}', alpha=0.6)

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=100, label='Centroids')

# Add titles and labels
plt.title('K-means Clustering of Customers (From Scratch)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
