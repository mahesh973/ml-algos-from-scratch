import numpy as np

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_index]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions


class AdaBoost:
    def __init__(self, n_estimators = 10):
        self.n_estimators = n_estimators

    def fit(self, X, y):
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
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.estimators]
        y_pred = np.sum(clf_preds, axis = 0)
        y_pred = np.sign(y_pred)
        return y_pred