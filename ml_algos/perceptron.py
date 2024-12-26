# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score

class Perceptron:
    """
    A simple implementation of the Perceptron algorithm for binary classification.

    Attributes:
        lr (float): Learning rate for weight updates.
        n_iters (int): Number of iterations for training.
        weights (np.ndarray): Weights vector for the model.
        bias (float): Bias term for the model.
        activation (callable): Activation function, set to a unit step function.
    """
    def __init__(self, learning_rate = 0.001, n_iters = 1000):
        """
        Initializes the Perceptron model with the given learning rate and number of iterations.

        Args:
            learning_rate (float): Learning rate for weight updates. Default is 0.001.
            n_iters (int): Number of iterations for training. Default is 1000.
        """
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.activation = self._unit_step_func

    def fit(self,X,y):
        """
        Trains the Perceptron model on the provided dataset.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray): True labels, shape (n_samples,). Assumes labels are 0 or 1.
        """
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_model = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_model)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights +=  update * x_i
                self.bias += update

    def predict(self,X):
        """
        Predicts the labels for the given dataset.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted labels, shape (n_samples,).
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation(linear_model)
        return y_predicted

    def _unit_step_func(self,x):
        """
        Unit step activation function.

        Args:
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Output values, where values >= 0 map to 1 and < 0 map to 0.
        """
        return np.where(x >= 0, 1, 0)



# Testing
if __name__ == "__main__":
    X, y = datasets.make_blobs(
        n_samples=1000, n_features=4, centers=2, cluster_std=1.05, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)

    print("Perceptron classification test set accuracy", accuracy_score(y_test, y_pred))