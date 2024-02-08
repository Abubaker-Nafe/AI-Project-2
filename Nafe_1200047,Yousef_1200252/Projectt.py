import numpy as np
import csv
import sys
import logging
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.spatial import distance

# Constants
TEST_SIZE = 0.3
K = 3
HIDDEN_LAYER_SIZES = (10, 5)
ACTIVATION = 'logistic'
MAX_ITER = 4601


class NN:
    """k-Nearest Neighbors classifier."""

    def __init__(self, training_features, training_labels):
        """Initialize the classifier with training data."""
        self.training_features = training_features
        self.training_labels = training_labels

    def predict(self, features, k):
        """Predict the labels for given features."""
        predictions = []
        for test_point in features:
            distances = [distance.euclidean(
                test_point, train_point) for train_point in self.training_features]
            indices = np.argsort(distances)[:k]
            labels = [self.training_labels[i] for i in indices]
            prediction = max(set(labels), key=labels.count)
            predictions.append(prediction)
        return predictions


def load_data(filename):
    """Load features and labels from a CSV file."""
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            features, labels = zip(
                *[(list(map(float, row[:-1])), int(row[-1])) for row in reader])
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
        sys.exit(1)
    return features, labels


def preprocess(features):
    """Normalize the features using z-score normalization."""
    features = np.array(features)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / std


def train_mlp_model(features, labels, hidden_layer_sizes, activation, max_iter):
    """Train a Multi-Layer Perceptron model."""
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                          activation=activation, max_iter=max_iter)
    model.fit(features, labels)
    return model


def evaluate(labels, predictions):
    """Evaluate the model using accuracy, precision, recall, and F1 score."""
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python template.py ./spambase.csv")

    features, labels = load_data(sys.argv[1])
    features = preprocess(features)
    training_features, test_features, training_labels, test_labels = train_test_split(
        features, labels, test_size=TEST_SIZE)

    model_nn = NN(training_features, training_labels)
    predictions_nnr = model_nn.predict(test_features, K)
    accuracy, precision, recall, f1 = evaluate(test_labels, predictions_nnr)

    print("**** 1-Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    confusion_matrix_nnr = confusion_matrix(test_labels, predictions_nnr)
    print("Confusion Matrix - 1-Nearest Neighbor (NNR):")
    print(confusion_matrix_nnr)
    print("\n")

    model = train_mlp_model(
        training_features, training_labels, HIDDEN_LAYER_SIZES, ACTIVATION, MAX_ITER)
    predictions_mlp = model.predict(test_features)
    accuracy, precision, recall, f1 = evaluate(test_labels, predictions_mlp)

    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    confusion_matrix_mlp = confusion_matrix(test_labels, predictions_mlp)
    print("Confusion Matrix - MLP:")
    print(confusion_matrix_mlp)


if __name__ == "__main__":
    main()
