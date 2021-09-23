import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from Per import Perceptron
from tqdm import tqdm


def main():
    # importing data
    df = pd.read_csv("F.data", delimiter=" ")
    df["T"] = df["C"] + df["M"] + df["X"]
    epochs = 100

    # extracting labels
    df["T"] = df["T"].apply(lambda y: -1 if y == 0 else 1)
    y = np.array(df["T"])

    # dropping labels
    df.drop(["T", "C", "X", "M"], axis=1, inplace=True)
    # converting letters to numbers #ascii
    for column in list(df.columns):
        df[column] = np.array(
            df[column].apply(lambda a: a if type(a) == int else (ord(a) - 65))
        )
    # feature vectors
    X = np.array(df)

    print(f"Training Vanilla")
    accuracies_std = train_test_accuracy(
        Perceptron(len(X[0])), X, y, epochs
    )

    print(f"Training Averaged")
    accuracies_avg = train_test_accuracy(
        Perceptron(len(X[0]), variation="avg"), X, y, epochs
    )

    plt.clf()
    plt.plot(range(epochs + 1), accuracies_std, label="Vanilla Perceptron")
    plt.plot(range(epochs + 1), accuracies_avg, label="Averaged Perceptron")
    plt.legend()
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title("Epoch vs Accuracy")
    plt.savefig("Epoch_vs_accuracy.png")


def train_test_accuracy(model, X, y, epochs):
    accuracies = []

    # train-test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=24, shuffle=True
    )

    # initial accuracy
    accuracies.append(model.calculate_accuracy(X_test, y_test) * 100)

    # epochs
    for epoch in tqdm(range(epochs)):
        # shuffling training data
        X_train, y_train = shuffle(X_train, y_train)
        # training
        model.train(X_train, y_train)
        # recording accuracy
        accuracies.append(model.calculate_accuracy(X_test, y_test) * 100)

    return accuracies


if __name__ == "__main__":
    main()