import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegression:
    """
    Linear Regression model.

    Attributes:
        lr (float): Learning rate for gradient descent.
        n_iters (int): Number of iterations for gradient descent.
        weights (np.ndarray): Weights of the model.
        bias (float): Bias term of the model.
    """
    def __init__(self, lr=0.001, n_iters=1000):
        """
        Initializes the LinearRegression model with learning rate and number of iterations.

        Args:
            lr (float): Learning rate for gradient descent. Default is 0.001.
            n_iters (int): Number of iterations for gradient descent. Default is 1000.
        """
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fits the LinearRegression model to the training data.

        Args:
            X (np.ndarray): Training data features.
            y (np.ndarray): Training data labels.
        """
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predicts the labels for the given data.

        Args:
            X (np.ndarray): Data features.

        Returns:
            np.ndarray: Predicted labels.
        """
        return np.dot(X, self.weights) + self.bias
    

if __name__ == "__main__":
    # Generate a dataset to test the LinearRegression implementation
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    print(f"The RMSE value on the test set is :{np.sqrt(mean_squared_error(y_test, y_pred))}")