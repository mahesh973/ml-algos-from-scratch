import numpy as np

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None
    
    def fit(self, X , y):
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
        return np.dot(X, self.linear_discriminants.T)