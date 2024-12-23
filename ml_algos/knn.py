# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # 1. Computing distances between x and all examples in the training set
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]

        # 2. Sorting by distance and returning indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]

        # 3. Extracting the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 4. Returning the most common class label
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common
    


if __name__ == "__main__":
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    def evaluate_knn_with_n_neighbors(n_neighbors):
        clf = KNN(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        return accuracy_score(y_test, predictions)
    
    accuracy_scores_with_n_neighbors = [evaluate_knn_with_n_neighbors(n_neighbors) for n_neighbors in range(1, X_train.shape[0])]
    
    # Plot the test set accuracy scores with varying n_neighbors
    plt.plot(range(1, X_train.shape[0]), accuracy_scores_with_n_neighbors)
    plt.xlabel("n_neighbors")
    plt.ylabel("Test set Accuracy")
    plt.title("Test Set Accuracy with varying n_neighbors")
    plt.show()



