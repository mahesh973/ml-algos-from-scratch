# Imports
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

class SVC:
    """
    Support Vector Classifier (SVC) implementing a simple linear SVM using gradient descent.

    Attributes:
        lr (float): Learning rate for gradient descent.
        lambda_param (float): Regularization parameter.
        n_iters (int): Number of iterations for training.
        w (ndarray): Weights of the model.
        b (float): Bias term of the model.
    """
    def __init__(self, learning_rate = 0.001, lambda_param = 0.01, n_iters = 1000):
        """
        Initialize the SVC with specified hyperparameters.

        Args:
            learning_rate (float): Learning rate for gradient descent.
            lambda_param (float): Regularization parameter.
            n_iters (int): Number of iterations for training.
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        Train the SVC using gradient descent.

        Args:
            X (ndarray): Feature matrix of shape (n_samples, n_features).
            y (ndarray): Target vector of shape (n_samples,).
                        Labels should be either -1 or 1.
        """
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.random.rand(n_features)
        self.b = np.random.rand(1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = (y_[idx] * (np.dot(x_i, self.w) - self.b)) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w) 
                    self.b -= 0
                else:
                    self.w -= self.lr * ((2 * self.lambda_param * self.w) - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        """
        Predict the class labels for the given input data.

        Args:
            X (ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            ndarray: Predicted labels (-1 or 1) for each sample.
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
    

# Testing
if __name__ == "__main__":
    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)

    clf = SVC()
    clf.fit(X, y)

    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    visualize_svm()