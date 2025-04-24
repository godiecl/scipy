#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.


def fizzbuzz(n: int):
    """
    Imprime los valores entre 1 y n:
    - Para multiplos de 3, imprime "Fizz"
    - Para multiplos de 5, imprime "Buzz"
    - Para multiplos de 3 y 5, imprime "FizzBuzz"
    """
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)


def main():
    fizzbuzz(100)


if __name__ == "__main__":
    main()
