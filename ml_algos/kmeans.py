import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KMeans:
    def __init__(self,k = 5, max_iters = 100, plot_steps = False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # initialize a list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.k)]

        # to store mean feature vector for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # 1. Initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimization
        for _ in range(self.max_iters):
            # 2. Assign each sample to the nearest centroid
            self.clusters = self._create_clusters(self.centroids)
            if self.plot_steps:
                self.plot()

            # 3. Re-compute the centroids
            old_centroids = self.centroids
            self.centroids = self._recompute_centroids(self.clusters)

            if self.plot_steps:
                self.plot()

            # 4. Check for convergence
            if self._is_converged(old_centroids, self.centroids):
                break

        # 5. Return cluster labels
        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]

        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def _recompute_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis = 0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, new_centroids):
        distances = [euclidean_distance(a,b) for a,b in zip(old_centroids, new_centroids)]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
    
    def plot(self):
        fig, ax = plt.subplots(figsize = (12,8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker = "x", color = 'black', linewidth=2)
        
        plt.show()


# Testing
if __name__ == "__main__":

    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(k=clusters, max_iters=150, plot_steps = True)
    y_pred = k.predict(X)

    print(y_pred)