from numpy import random, dot, array, append, round, average
from sklearn.metrics import confusion_matrix


class Perceptron:
    def __init__(self, feature_vector_length, learning_rate=0.001, variation="std"):
        self.weights = random.rand(feature_vector_length + 1)
        self.learning_rate = learning_rate
        self.variation = variation
        if self.variation == "avg":
            self.weights_archive = [self.weights]

    def train(self, X, y):
        X = array(list(map(lambda x: append(array([1]), x), X)))
        for x, y_ in zip(X, y):
            y_predict = dot(x, self.weights)
            self.weights = self.weights + self.learning_rate * (y_ - y_predict) * x
            if self.variation == "avg":
                self.weights_archive.append(self.weights)
        return self

    def fit_batch(self, X, y):
        X = array(list(map(lambda x: append(array([1]), x), X)))
        y_dif = []
        for x, y_ in zip(X, y):
            y_dif.append((y_ - dot(x, self.weights)) * x)
        self.weights = self.weights + self.learning_rate * sum(y_dif) / len(X)

    def fit_epochs(
        self,
        X,
        y,
        epoch_length,
    ):
        epoch_length = int(epoch_length)
        for i in range(int(len(X) / epoch_length)):
            self.fit_batch(X[i : i + epoch_length], y[i : i + epoch_length])
        return self

    def to_class(self, l):
        return array(list(map(lambda x: -1 if x < 0 else 1, l)))

    def predict(self, X):
        X = array(list(map(lambda x: append(array([1]), x), X)))
        if self.variation == "std":
            weights = self.weights
        elif self.variation == "avg":
            weights = average(array(self.weights_archive).T, axis=1)
        predicted_values = list(map(lambda x: dot(x, weights), X))
        return self.to_class(predicted_values)

    def calculate_accuracy(self, X, y):
        y_predict = self.predict(X)

        total = 0
        correct = 0
        for y_t, y_p in zip(y, y_predict):
            total += 1
            if y_t == y_p:
                correct += 1

        return correct / total