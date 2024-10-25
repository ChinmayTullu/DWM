import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Compute the minimum pairwise distance between two clusters
def compute_distance_matrix(clusters):
    n = len(clusters)
    dist_matrix = np.full((n, n), float('inf'))  
    for i in range(n):
        for j in range(i + 1, n):
            # Convert lists to NumPy arrays for calculations
            dist_matrix[i, j] = min(np.linalg.norm(np.array(p1) - np.array(p2)) for p1 in clusters[i] for p2 in clusters[j])
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

# Perform single-linkage agglomerative clustering
def single_linkage_clustering(data):
    clusters = [[point] for point in data]
    labels = list(range(len(clusters)))
    merge_history = []
    current_label = len(labels)  # Start unique labels from the current number of points
    
    while len(clusters) > 1:
        # Compute the distance matrix
        dist_matrix = compute_distance_matrix(clusters)
        # Find the pair of clusters with the minimum distance
        c1, c2 = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        # Record the merge history [cluster1, cluster2, distance, new cluster size]
        merge_history.append([labels[c1], labels[c2], dist_matrix[c1, c2], len(clusters[c1]) + len(clusters[c2])])
        # Merge clusters c1 and c2
        clusters[c1].extend(clusters[c2])
        del clusters[c2]  # Remove the merged cluster
        labels[c1] = current_label  # Assign a new unique label for the merged cluster
        del labels[c2]  # Remove the label of the merged cluster
        current_label += 1

    return merge_history

# Plot dendrogram
def plot_dendrogram(merge_history):
    linkage_matrix = np.array(merge_history)
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title("Dendrogram (Single Linkage)")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()

# Load and cluster static data
def main():
    # Static data points for clustering
    data = [
        [12, 13],
        [1, 2],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7],
        [8, 9],
        [10, 11],
        [12, 14],
        [13, 15]
    ]
    
    # Perform clustering
    merge_history = single_linkage_clustering(data)
    
    # Plot the dendrogram
    plot_dendrogram(merge_history)

if __name__ == "__main__":
    main()
