# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class PCA:
    def __init__(self, n_components = None):
        self.n_components = n_components
        self.components = None
        self.mean = None

    
    def fit(self, X):
        # 1. Subtract the mean from the data
        self.mean = np.mean(X, axis = 0)
        X -= self.mean

        # 2. Calculate the covariance matrix
        cov_matrix = np.cov(X, rowvar = False)

        # 3. Calculate eigenvectors and eigenvalues of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

        # 4. Sort the eigenvectors according to the eigenvalues in decreasing order
        eigen_vectors = eigen_vectors.T

        indexes = np.argsort(eigen_values)[::-1]
        eigen_vectors = eigen_vectors[indexes]
        eigen_values = eigen_values[indexes]

        # 5. Choose first k eigenvectors, where k is the number of components
        self.components = eigen_vectors[:self.n_components]

    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T) 
    

# Testing
if __name__ == "__main__":
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project the data onto the 2 primary principal components
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    plt.show()