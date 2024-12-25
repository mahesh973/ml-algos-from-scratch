import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


class LogisticRegression:
    """
    A simple implementation of Logistic Regression using gradient descent.

    Attributes:
        lr (float): Learning rate for the gradient descent optimization.
        n_iters (int): Number of iterations to run gradient descent.
        weights (numpy.ndarray): Coefficients for the features after training.
        bias (float): Bias term after training.
    """
    def __init__(self,lr = 0.001, n_iters = 1000):
        """
        Initializes the LogisticRegression model with specified learning rate and number of iterations.

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
        Trains the Logistic Regression model using gradient descent.

        Args:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
            y (numpy.ndarray): Target vector of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features) 
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        Predicts binary class labels for the given feature matrix.

        Args:
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            list: Predicted class labels (0 or 1) for each sample.
        """
        y_predicted = self._sigmoid(np.dot(X, self.weights) + self.bias)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self,x):
        """
        Computes the sigmoid function.

        Args:
            x (numpy.ndarray or float): Input value(s).

        Returns:
            numpy.ndarray or float: Sigmoid of the input value(s).
        """
        return 1 / (1 + np.exp(-x))
    

if __name__ == "__main__":
    # Generate a dataset to test the LogisticRegression implementation
    X, y = datasets.make_classification(n_samples=1000, n_features=4, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    classifier = LogisticRegression(lr = 0.01, n_iters = 1000)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(f"The accuracy on the test set is :{accuracy_score(y_test, y_pred)}")