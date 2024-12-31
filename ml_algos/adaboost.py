# Imports
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DecisionStump:
    """
    A simple decision stump (weak classifier) used in AdaBoost.
    
    Attributes
    ----------
    polarity : int
        Indicates the direction of the inequality (1 for <, -1 for >).
    
    feature_index : int
        The index of the feature used for splitting.
    
    threshold : float
        The threshold value for the feature to make predictions.
    
    alpha : float
        The weight of the stump in the final AdaBoost classifier.
    """
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        """
        Make predictions using the decision stump.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted labels (-1 or 1) for each sample.
        """
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


class AdaBoost:
    """
    AdaBoost ensemble classifier using decision stumps as weak classifiers.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of weak classifiers (decision stumps) to use in the ensemble.

    Attributes
    ----------
    estimators : list
        A list of trained decision stumps.
    """
    def __init__(self, n_estimators = 10):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        """
        Train the AdaBoost classifier on the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix.

        y : ndarray of shape (n_samples,)
            The target labels (-1 or 1).
        """
        n_samples, n_features = X.shape
        
        # Initialize weights
        weights = np.full(n_samples, (1/n_samples))

        self.estimators = []
        for _ in range(self.n_estimators):
            clf = DecisionStump()

            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    misclassified = weights[y != predictions]
                    error = sum(misclassified)

                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + EPS))

            predictions = clf.predict(X)

            weights *= np.exp(-clf.alpha * y * predictions)
            weights /= np.sum(weights)

            self.estimators.append(clf)


    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix for which predictions are to be made.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (-1 or 1) for each sample.
        """
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.estimators]
        y_pred = np.sum(clf_preds, axis = 0)
        y_pred = np.sign(y_pred)
        return y_pred
    

# Testing
if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    y[y == 0] = -1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # AdaBoost classification with 10 weak classifiers
    clf = AdaBoost(n_estimators=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("Test set Accuracy:", acc)