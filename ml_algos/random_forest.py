    # Imports
import numpy as np
from decision_tree import DecisionTreeClassifier
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def bootstrap_sample(X, y, n_features = 'sqrt'):
    """
    Generate a bootstrap sample from the dataset and select a subset of features.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The feature matrix.

    y : ndarray of shape (n_samples,)
        The target labels.

    n_features : int or str, default='sqrt'
        The number of features to select for each tree. If 'sqrt', selects the 
        square root of the total number of features.

    Returns
    -------
    X_sample : ndarray of shape (n_samples, n_selected_features)
        The feature matrix for the bootstrap sample with selected features.

    y_sample : ndarray of shape (n_samples,)
        The target labels for the bootstrap sample.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    idxs = np.random.choice(n_samples, size = n_samples, replace=True)
    features = np.random.choice(n_features, size = int(np.round(np.sqrt(n_features))), replace=False)
    return X[idxs, :][:,features], y[idxs]

def most_common_label(y):
    """
    Determine the most common label in a set of labels.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        The target labels.

    Returns
    -------
    int
        The most common label.
    """
    counter = Counter(y)
    return counter.most_common(1)[0][0]


class RandomForestClassifier:
    """
    A Random Forest classifier, which builds an ensemble of decision trees 
    using bootstrapped datasets and feature subsampling.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.

    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.

    max_depth : int, default=100
        The maximum depth of the tree.

    n_feats : int, optional, default=None
        The number of features to consider when looking for the best split. 
        If None, all features are considered.
    """
    def __init__(self, n_estimators = 100, min_samples_split = 2, max_depth = 100, n_feats = None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        """
        Fit the random forest classifier by training multiple decision trees on bootstrapped samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix.

        y : ndarray of shape (n_samples,)
            The target labels.
        """
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(min_samples_split = self.min_samples_split,
                                          max_depth = self.max_depth,
                                          n_feats = self.n_feats)
            
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict the class labels for the input samples by aggregating the predictions from all trees.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The feature matrix for which predictions are to be made.

        Returns
        -------
        y_pred : list of int
            The predicted class labels for each sample in X.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)

        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return y_pred

# Testing
if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=50, max_depth=5)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Test set Accuracy:", acc)