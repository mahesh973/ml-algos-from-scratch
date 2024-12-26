# Imports
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayesClassifier:
    """
    Naive Bayes classifier for continuous data using Gaussian distribution.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fits the Naive Bayes model to the training data.

        Parameters:
        X (ndarray): Feature matrix of shape (n_samples, n_features).
        y (ndarray): Target vector of shape (n_samples,).

        Returns:
        None
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Initialize mean, variance and priors
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64)
        self._var = np.zeros((n_classes, n_features), dtype = np.float64)
        self._priors = np.zeros(n_classes, dtype = np.float64)

        for idx, c in enumerate(self._classes):
            X_c =  X[y == c]
            self._mean[idx, :] = X_c.mean(axis = 0)
            self._var[idx, :] = X_c.var(axis = 0)
            self._priors[idx] = X_c.shape[0] / n_samples

    def predict(self, X):
        """
        Predicts the class labels for the input data.

        Parameters:
        X (ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
        ndarray: Predicted class labels of shape (n_samples,).
        """
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        """
        Predicts the class label for a single sample.

        Parameters:
        x (ndarray): Feature vector of shape (n_features,).

        Returns:
        int: Predicted class label.
        """
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        """
        Computes the probability density function of the Gaussian distribution 
        for a given class and sample.

        Parameters:
        class_idx (int): Index of the class.
        x (ndarray): Feature vector of shape (n_features,).

        Returns:
        ndarray: Computed probabilities of shape (n_features,).
        """
        mean = self._mean[class_idx, :]
        var = self._var[class_idx, :]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


# Testing
if __name__ == "__main__":
    X, y = datasets.make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify= y
    )

    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    print("Naive Bayes test set classification accuracy", accuracy_score(y_test, y_pred))