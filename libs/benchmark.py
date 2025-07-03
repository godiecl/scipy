#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import datetime
import logging
import time
from contextlib import contextmanager
from logging import Logger
from typing import Generator, Optional

import humanize
from typeguard import typechecked


@contextmanager
@typechecked
def benchmark(
    operation_name: Optional[str] = None,
    log: Optional[Logger] = None,
) -> Generator[None, None, None]:
    """Function to benchmark a function."""

    # time calculated in nanoseconds
    start: int = time.perf_counter_ns()
    try:
        # yield control to the caller
        yield
    finally:
        # the elapsed time
        elapsed_ns: int = time.perf_counter_ns() - start

        # convert to microseconds
        elapsed_ms = elapsed_ns / 1_000

        # the variation
        elapsed_human = humanize.precisedelta(
            value=datetime.timedelta(microseconds=elapsed_ms),
            minimum_unit="microseconds",
            format="%.2f",
        )

        # if no logger is provided, use the default logger
        if log is None:
            log = logging.getLogger(__name__)

        # if no operation name is provided, use the default
        if operation_name is None:
            log.debug(f"* executed in {elapsed_human}.")
        else:
            log.debug(f"* {operation_name} executed in {elapsed_human}.")
