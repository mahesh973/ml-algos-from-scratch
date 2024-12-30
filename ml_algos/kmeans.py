import numpy as np
np.random.seed(42)

def euclidean_distance(x1,x2):
    np.sqrt(np.sum(x1 - x2)**2)

class KMeans:
    def __init__(self, k=5, max_iters = 1000):
        self.k = k
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def fit_transform(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimization
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)


    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
             centroid_idx = self._closest_centroid(sample, centroids)
             clusters[centroid_idx].append(idx)
    
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx