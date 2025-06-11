#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import math

from typeguard import typechecked


@typechecked
def is_prime(n: int) -> bool:
    """
    Check if a number is prime.
    """
    if n <= 1:
        return False

    if n == 2:
        return True

    if n % 2 == 0:
        return False

    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False

    return True
