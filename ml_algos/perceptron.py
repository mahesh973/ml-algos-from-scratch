# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score

class Perceptron:
    def __init__(self, learning_rate = 0.001, n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.activation = self._unit_step_func

    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_model = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation(linear_model)

                update = self.lr * (y[idx] - y_predicted)
                self.weights +=  update * x_i
                self.bias += update

    def predict(self,X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation(linear_model)
        return y_predicted

    def _unit_step_func(self,x):
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