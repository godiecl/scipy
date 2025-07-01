#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging
import os
from typing import Sequence

from prettytable import PrettyTable

from logger import configure_logging
from sqlalchemy import Engine, Row, TextClause, create_engine, text
from typeguard import typechecked


@typechecked
def execute_sql(sql: TextClause, engine: Engine) -> Sequence[Row]:
    log = logging.getLogger(__name__)
    log.debug(f"Executing sql: {sql}.")

    try:
        with engine.connect() as connection:
            # execute the sql
            result = connection.execute(sql)
            # retrieve all the rows
            rows = result.fetchall()
            log.debug(f"Query executed ok, returned {len(rows)} rows.")
            return rows
    except Exception as e:
        log.error(f"Error executing sql: {e}")
        raise e


@typechecked
def main():
    log = logging.getLogger(__name__)
    log.info("Starting ..")

    # the data dir
    data_dir = os.path.join("..", "..", "data")

    # the database file
    db_file = os.path.join(data_dir, "flights.sqlite")

    # creating the sqlalchemy engine
    engine = create_engine(f"sqlite:///{db_file}")

    average_arrival_and_departure_by_month = text("""
                                                  -- Average arrival and departure delays by month
                                                  SELECT month,
                                                         ROUND(AVG(arrdelay), 2) AS avg_arrdelay,
                                                         ROUND(AVG(depdelay), 2) AS avg_depdelay
                                                  FROM flights
                                                  GROUP BY month
                                                  ORDER BY avg_depdelay;""")

    aaadbm = execute_sql(average_arrival_and_departure_by_month, engine)

    table = PrettyTable()
    table.field_names = ["Month", "Average Arrive Delay", "Average Departure Delay"]
    for row in aaadbm:
        table.add_row([row[0], row[1], row[2]])

    log.debug(f"Table: Average arrival and departure delays by month:\n{table}")

    log.info("Done.")


if __name__ == "__main__":
    configure_logging(log_level="DEBUG")
    main()
