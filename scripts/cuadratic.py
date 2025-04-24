#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import math
import cmath
from typing import Tuple


def cuadratic(
    a: float, b: float, c: float
) -> Tuple[float, float] | Tuple[complex, complex]:
    """
    Funcion que calcula las raices cuadraticas dado los valores de a, b y c
    ax^2 + bx + c = 0
    """

    if a == 0:
        raise ValueError("a cannot be zero")

    discriminante = b**2 - 4 * a * c

    if discriminante >= 0:
        root_1 = (-b + math.sqrt(discriminante)) / (2 * a)
        root_2 = (-b - math.sqrt(discriminante)) / (2 * a)
    else:
        root_1 = (-b - cmath.sqrt(discriminante)) / (2 * a)
        root_2 = (-b + cmath.sqrt(discriminante)) / (2 * a)

    return root_1, root_2


def main():
    # lectura de los coeficientes
    a = float(input("Ingrese el coeficiente a: "))
    b = float(input("Ingrese el coeficiente b: "))
    c = float(input("Ingrese el coeficiente c: "))

    # calculo de las raices
    root_1, root_2 = cuadratic(a, b, c)
    print(f"Las raices son: {root_1} y {root_2}")


if __name__ == "__main__":
    main()
