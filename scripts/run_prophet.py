#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging

import pandas as pd
from matplotlib import pyplot as plt
from prophet import Prophet

from benchmark import benchmark
from logger import configure_logging


def main():
    log = logging.getLogger(__name__)
    log.info("Starting ..")

    # retrieve the data
    log.debug("Retrieving the data from the web ...")
    with benchmark("Dowload data ..", log):
        df = pd.read_csv(
            "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
        )
    log.debug(f"DataFrame shape: {df.shape}")
    log.debug(f"Data: {df.head()}")

    # the model
    # model = Prophet()
    model = Prophet(
        changepoints=["2011-01-01", "2013-01-01"],
    )
    with benchmark("Fit()", log):
        model.fit(df)

    # create the future dataframe
    future = model.make_future_dataframe(periods=365)

    # predict the future
    forecast = model.predict(future)

    # plot the forecast
    model.plot(forecast)
    plt.show()

    # plot the components
    model.plot_components(forecast)
    plt.show()

    log.info("Done.")


if __name__ == "__main__":
    configure_logging(log_level="DEBUG")
    main()
