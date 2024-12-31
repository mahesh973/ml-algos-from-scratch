# Imports
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

def entropy(y):
    """
    Compute the entropy of a label distribution.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Array of target labels.

    Returns
    -------
    float
        Entropy of the label distribution.
    """
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    """
    A single node in the Decision Tree.

    Parameters
    ----------
    feature : int, default=None
        Index of the feature used for splitting the data.

    threshold : float, default=None
        Threshold value for the feature split.

    left : Node, default=None
        Left child node.

    right : Node, default=None
        Right child node.

    value : int, default=None
        Class label for a leaf node. If None, the node is not a leaf.

    Attributes
    ----------
    feature : int
        Index of the feature used for splitting.

    threshold : float
        Threshold value for splitting.

    left : Node
        Left child node.

    right : Node
        Right child node.

    value : int
        Class label of the leaf node if the node is a leaf; otherwise, None.
    """
    def __init__(self, feature = None, threshold = None, left = None, right = None, * ,value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Check if the node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf, otherwise False.
        """
        return self.value is not None

class DecisionTreeClassifier:
    """
    Decision Tree Classifier using the CART algorithm.

    Parameters
    ----------
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.

    max_depth : int, default=100
        Maximum depth of the tree.

    n_feats : int or None, default=None
        Number of features to consider when looking for the best split.
        If None, all features are considered.

    Attributes
    ----------
    root : Node
        The root node of the decision tree.
    """
    def __init__(self, min_samples_split=2, max_depth = 100, n_feats = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y): 
        """
        Build the decision tree classifier from the training set.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        y : ndarray of shape (n_samples,)
            The target labels.
        """
        # Grow the tree
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)
    
    def predict(self, X):
        """
        Predict class labels for the input samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels for each sample.
        """
        # Traverse the tree
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _grow_tree(self, X, y, depth = 0):
        """
        Recursively grow the decision tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        y : ndarray of shape (n_samples,)
            The target labels.

        depth : int, default=0
            Current depth of the tree.

        Returns
        -------
        Node
            A Node object representing the root of the grown tree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # 1. Stopping Criteria
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # 2. Continue growing the tree
        feature_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # Greedy Search
        best_feature, best_threshold = self._best_criteria(X, y, feature_idxs)
        left_idxs, right_idxs = self._split(X[:,best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_threshold, left, right)

    def _best_criteria(self, X, y, feature_idxs):
        """
        Find the best feature and threshold for splitting.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        y : ndarray of shape (n_samples,)
            The target labels.

        feature_idxs : ndarray of shape (n_selected_features,)
            Indices of features to consider for the split.

        Returns
        -------
        tuple
            Index of the best feature and the best threshold value.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold
    
    def _traverse_tree(self, x, node):
        """
        Traverse the tree to predict the label for a single sample.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            A single input sample.

        node : Node
            The current node in the decision tree.

        Returns
        -------
        int
            Predicted class label for the sample.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _information_gain(self, y, X_column, split_threshold):
        """
        Compute the information gain for a split.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            The target labels.

        X_column : ndarray of shape (n_samples,)
            Column of feature values.

        split_threshold : float
            Threshold value for the split.

        Returns
        -------
        float
            Information gain for the split.
        """
        # 1. Calculate parent Entropy
        parent = entropy(y)

        # 2. Generate the split
        left_idxs, right_idxs = self._split(X_column, split_threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # 3. Calculate weighted average of child E
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # 4. Return Information gain
        information_gain = parent - child_entropy
        return information_gain

    def _split(self, X_column, split_threshold):
        """
        Split the data based on a feature threshold.

        Parameters
        ----------
        X_column : ndarray of shape (n_samples,)
            Column of feature values.

        split_threshold : float
            Threshold value for the split.

        Returns
        -------
        tuple
            Indices of samples in the left and right splits.
        """
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()
        right_idxs = np.argwhere(X_column > split_threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
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


if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTreeClassifier(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)