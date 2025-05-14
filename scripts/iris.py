#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging

import pandas as pd
from benchmark import benchmark
from logger import configure_logging
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix


def main():
    # configure the logger
    configure_logging(logging.DEBUG)

    # create a logger
    log = logging.getLogger(__name__)
    log.debug("Starting the main ..")

    with benchmark(
        operation_name="read_csv",
        log=log,
    ):
        # reading the data
        df = pd.read_csv("../data/iris.csv")
        log.debug(f"Data readed: {df}")

    # print the first 5 rows
    log.debug(f"head:\n{df.head()}")

    # print the last 5 rows
    log.debug(f"tail:\n{df.tail()}")

    # describe the data
    log.debug(f"describe:\n{df.describe()}")

    # print the shape
    log.debug(f"info:\n{df.info()}")

    # log.debug("Profiling ..")
    # profile = ProfileReport(df, title="Iris Dataset Profiling Report")
    # profile.to_file("../output/iris_report.html")

    # graphing the data
    scatter_matrix(df[df["variety"] == "Iris-setosa"], c="red", label="Iris-setosa")

    # show!
    plt.show()

    log.debug("Done.")


if __name__ == "__main__":
    main()
