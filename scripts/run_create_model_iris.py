#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging
import pickle

import pandas as pd
from benchmark import benchmark
from logger import configure_logging
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree


def visualize_tree(model: RandomForestClassifier, x, y):
    # get the internal tree from the model
    tree = model.estimators_[0]

    plt.figure(figsize=(20, 10))

    plot_tree(
        tree,
        feature_names=x.columns.tolist(),
        class_names=model.classes_.tolist(),
        filled=True,
        rounded=True,
        fontsize=12,
    )

    plt.show()


def main():
    log = logging.getLogger(__name__)
    log.info("Starting the model creation process for the Iris dataset...")

    # read the Iris dataset
    df_iris = pd.read_csv("../data/iris.csv")
    log.info(f"Iris dataset loaded successfully:\n{df_iris.head()}")

    # the characteristics (x) and the objective variable (y)
    x = df_iris[["sepal.length", "sepal.width", "petal.length", "petal.width"]]
    y = df_iris["variety"]

    # split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    log.debug(f"Training dataset: {len(x_train)} samples.")
    log.debug(f"Testing dataset: {len(x_test)} samples.")

    # train the model
    model = RandomForestClassifier(random_state=42, max_depth=2)
    with benchmark("Training the Random Forest model", log):
        model.fit(x_train, y_train)
    log.debug("Model training completed successfully.")

    # evaluate the model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    log.debug(f"Model accuracy: {accuracy:.3f}")

    # visualize the first tree in the forest
    # visualize_tree(model, x_train, y_train)

    # save the model
    with open("../output/model_iris_randomforest.pkl", "wb") as f:
        pickle.dump(model, f)

    log.info("Done.")


if __name__ == "__main__":
    configure_logging(log_level="DEBUG")
    main()
