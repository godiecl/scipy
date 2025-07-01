#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging
import os

import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm

from benchmark import benchmark
from logger import configure_logging


def main():
    log = logging.getLogger(__name__)
    log.info("Starting ..")

    # the data dir
    data_dir = os.path.join("..", "..", "data")

    # the flight file
    csv_file = os.path.join(data_dir, "2008.csv")

    # the database file
    db_file = os.path.join(data_dir, "flights.sqlite")

    # Number of rows per chunk
    chunk_size = 100000

    # creating the sqlalchemy engine
    engine = create_engine(f"sqlite:///{db_file}")

    with benchmark("Loaded flights into the database.", log):
        for i, chunk in tqdm(enumerate(pd.read_csv(csv_file, chunksize=chunk_size))):
            # log.debug(f"Loading chunk {i} into the database ..")
            chunk.columns = [col.lower() for col in chunk.columns]
            with engine.begin() as connection:
                chunk.to_sql(
                    "flights", con=connection, if_exists="replace", index=False
                )

    log.info("Done.")


if __name__ == "__main__":
    configure_logging(log_level="DEBUG")
    main()
