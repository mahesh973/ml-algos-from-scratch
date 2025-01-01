# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class LDA:
    """
    Linear Discriminant Analysis (LDA) for dimensionality reduction.

    Parameters
    ----------
    n_components : int
        Number of linear discriminants to project the data onto.

    Attributes
    ----------
    linear_discriminants : ndarray of shape (n_components, n_features)
        The matrix of linear discriminants for the transformation.
    """
    def __init__(self, n_components):
        """
        Initialize the LDA model with the desired number of components.

        Parameters
        ----------
        n_components : int
            Number of linear discriminants to retain.
        """
        self.n_components = n_components
        self.linear_discriminants = None
    
    def fit(self, X , y):
        """
        Fit the LDA model to the data by finding the linear discriminants.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix.
        
        y : ndarray of shape (n_samples,)
            The target labels.

        Returns
        -------
        None
        """
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Initialize scatter matrices for both between and within classes
        mean_overall = np.mean(X, axis = 0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis = 0)
            S_W += np.dot((X_c - mean_c).T, (X_c - mean_c))

            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(-1,1)
            S_B += (n_c * np.dot(mean_diff, mean_diff.T))
        
        A = np.dot(np.linalg.inv(S_W) , S_B)

        eig_values, eig_vectors = np.linalg.eig(A)
        eig_vectors = eig_vectors.T
        idxs = np.argsort(abs(eig_values))[::-1]
        eig_values, eig_vectors = eig_values[idxs], eig_vectors[idxs]
        self.linear_discriminants = eig_vectors[:self.n_components]


    def transform(self, X):
        """
        Project the data onto the linear discriminants.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix to be transformed.

        Returns
        -------
        X_projected : ndarray of shape (n_samples, n_components)
            The data projected onto the top linear discriminants.
        """
        return np.dot(X, self.linear_discriminants.T)
    

# Testing
if __name__ == "__main__":
    data = datasets.load_iris()
    X, y = data.data, data.target

    # Project the data onto the 2 primary linear discriminants
    lda = LDA(2)
    lda.fit(X, y)
    X_projected = lda.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1, x2 = X_projected[:, 0], X_projected[:, 1]

    plt.scatter(
        x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
    )

    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
    plt.colorbar()
    plt.show()