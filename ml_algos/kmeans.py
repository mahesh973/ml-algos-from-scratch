import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
np.random.seed(42)

def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two points.

    Parameters
    ----------
    x1 : ndarray of shape (n_features,)
        The first point.

    x2 : ndarray of shape (n_features,)
        The second point.

    Returns
    -------
    float
        The Euclidean distance between `x1` and `x2`.
    """
    return np.sqrt(np.sum((x1 - x2)**2))


class KMeans:
    """
    K-Means clustering algorithm.

    Parameters
    ----------
    k : int, default=5
        The number of clusters.

    max_iters : int, default=100
        Maximum number of iterations for the clustering algorithm.

    plot_steps : bool, default=False
        If True, plot the steps of the algorithm at each iteration.

    Attributes
    ----------
    clusters : list of list
        Indices of samples in each cluster.

    centroids : list of ndarray of shape (k, n_features)
        Coordinates of the cluster centroids.
    """
    def __init__(self,k = 5, max_iters = 100, plot_steps = False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # initialize a list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.k)]

        # to store mean feature vector for each cluster
        self.centroids = []

    def predict(self, X):
        """
        Fit the K-Means model and predict the cluster for each sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to cluster.

        Returns
        -------
        ndarray of shape (n_samples,)
            Cluster labels for each sample.
        """
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
        """
        Assign each sample to the nearest centroid.

        Parameters
        ----------
        centroids : list of ndarray of shape (k, n_features)
            Current centroids of the clusters.

        Returns
        -------
        list of list
            Indices of samples in each cluster.
        """
        clusters = [[] for _ in range(self.k)]

        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self, sample, centroids):
        """
        Identify the index of the nearest centroid to a given sample.

        Parameters
        ----------
        sample : ndarray of shape (n_features,)
            Sample for which the closest centroid is to be determined.

        centroids : list of ndarray of shape (k, n_features)
            Current centroids of the clusters.

        Returns
        -------
        int
            Index of the nearest centroid.
        """
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def _recompute_centroids(self, clusters):
        """
        Update centroids as the mean of samples assigned to each cluster.

        Parameters
        ----------
        clusters : list of list
            Indices of samples in each cluster.

        Returns
        -------
        ndarray of shape (k, n_features)
            Updated centroids.
        """
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis = 0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, old_centroids, new_centroids):
        """
        Check if the algorithm has converged.

        Parameters
        ----------
        old_centroids : list of ndarray of shape (k, n_features)
            Centroids from the previous iteration.

        new_centroids : list of ndarray of shape (k, n_features)
            Centroids from the current iteration.

        Returns
        -------
        bool
            True if centroids have not changed, otherwise False.
        """
        distances = [euclidean_distance(a,b) for a,b in zip(old_centroids, new_centroids)]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        """
        Generate cluster labels for each sample.

        Parameters
        ----------
        clusters : list of list
            Indices of samples in each cluster.

        Returns
        -------
        ndarray of shape (n_samples,)
            Cluster labels for each sample.
        """
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
    
    def plot(self):
        """
        Visualize the current clustering state.

        Shows the clusters and centroids using a scatter plot.
        """
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

    k = KMeans(k=clusters, max_iters=150, plot_steps = True)
    y_pred = k.predict(X)
    k.plot()