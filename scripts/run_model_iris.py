#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging
import pickle
from pathlib import Path

import pandas as pd
import streamlit
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from logger import configure_logging


@streamlit.cache_data
def load_iris_model() -> RandomForestClassifier:
    model_path = Path("output/model_iris_randomforest.pkl")
    with model_path.open("rb") as model_file:
        return pickle.load(model_file)


def main():
    log = logging.getLogger(__name__)
    log.info("Starting the web service for the Iris model...")

    # load the model from the file system
    model = load_iris_model()

    # the webpage
    streamlit.title("Iris Flower Classification")
    streamlit.markdown(
        "This web service uses a **Random Forest** model to classify Iris flowers based on their features."
    )
    streamlit.header("Iris Flower Features")
    col_1, col_2 = streamlit.columns(2)

    with col_1:
        streamlit.text("Sepal:")
        sepal_lenght = streamlit.slider(
            "Select Sepal Length",
            min_value=4.0,
            max_value=8.0,
            value=5.0,
            step=0.1,
        )
        sepal_width = streamlit.slider(
            "Select Sepal Width",
            min_value=2.0,
            max_value=5.0,
            value=3.0,
            step=0.1,
        )

    with col_2:
        streamlit.text("Petal:")
        petal_length = streamlit.slider(
            "Select Petal Length",
            min_value=1.0,
            max_value=7.0,
            value=1.5,
            step=0.1,
        )
        petal_width = streamlit.slider(
            "Select Petal Width",
            min_value=0.1,
            max_value=2.5,
            value=0.2,
            step=0.1,
        )

    # create a DataFrame with the features
    features = pd.DataFrame(
        [[sepal_lenght, sepal_width, petal_length, petal_width]],
        columns=["sepal.length", "sepal.width", "petal.length", "petal.width"],
    )

    # make a prediction
    predict = model.predict(features)[0]
    streamlit.write(f"The predicted variety of the Iris flower is: **{predict}**")

    # predict probabilities
    probabilities = model.predict_proba(features)[0]

    probabilities_df = pd.DataFrame(
        probabilities, index=model.classes_, columns=["Probability"]
    )

    # draw the probabilities
    plt.figure(figsize=(8, 4))

    plt.title("Predicted Probabilities for Iris Flower Varieties")
    probabilities_df.plot(kind="bar", legend=False, ax=plt.gca())
    plt.tight_layout()

    streamlit.pyplot(plt)

    log.info("Done.")


if __name__ == "__main__":
    configure_logging(log_level="DEBUG")
    main()
