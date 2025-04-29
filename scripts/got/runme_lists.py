#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.

"""
Implementacion del Juego de la Vida.

Este modulo implementa el juego de la vida basado en un automata celular ideado
por Don John Conway.

En cada celda se puede encontrar una celula que podria estar viva (1) o muerta
(0) dependiendo de los vecinos que tienen.

https://es.wikipedia.org/wiki/Juego_de_la_vida
"""


def imprimir_estado(estado):
    for fila in estado:
        print("".join("࿕" if celula == 1 else " " for celula in fila))


def crear_nuevo(estado):
    # tamanio de la matriz estado
    n_filas = len(estado)
    n_columnas = len(estado[0])

    # creo una nueva matriz del mismo tamanio que el estado original
    nuevo_estado = []
    for f in range(n_filas):
        # construyo una nueva fila
        nueva_fila = []
        for c in range(n_columnas):
            nueva_fila.append(0)
        nuevo_estado.append(nueva_fila)
    return nuevo_estado


def contar_vecinos(estado, fila, columna):
    contador_vecinos = 0

    # ciclo para recorrer las filas
    for f in range(fila - 1, fila + 2):
        # ciclo para recorrer las columnas
        for c in range(columna - 1, columna + 2):
            # no se cuenta la propia celula
            if f == fila and c == columna:
                continue
            # estoy fuera de la matriz por fila
            if f < 0 or f >= len(estado):
                continue
            # estoy fuera de la matriz por columna
            if c < 0 or c >= len(estado[0]):
                continue

            # hay un vecino -> lo cuento
            if estado[f][c] == 1:
                contador_vecinos += 1

    return contador_vecinos


def expandir(estado):
    """
    Function que agrega una fila y una columna al inicio y al final de la matriz.
    """
    # tamanio de la matriz estado
    n_filas = len(estado)
    n_columnas = len(estado[0])

    # creo una nueva matriz del mismo tamanio que el estado original
    nuevo_estado = []
    for f in range(n_filas + 2):
        # construyo una nueva fila
        nueva_fila = []
        for c in range(n_columnas + 2):
            nueva_fila.append(0)
        nuevo_estado.append(nueva_fila)

    # inserto la matriz original en el centro de la nueva
    for f in range(n_filas):
        for c in range(n_columnas):
            nuevo_estado[f + 1][c + 1] = estado[f][c]

    return nuevo_estado


def comprimir(estado):
    # se realizo algun cambio?
    cambios = True
    while cambios:
        cambios = False

        # guardar dimensiones actuales para detectar cambios
        filas_antes = len(estado)
        if filas_antes == 0:
            return estado
        columnas_antes = len(estado[0])

        # elimina la primera fila si contiene solo ceros
        if all(celda == 0 for celda in estado[0]):
            estado = estado[1:]
            cambios = True

        # verificar si el estado esta vacio después de eliminar filas
        if not estado:
            return estado

        # elimina la ultima fila si contiene solo ceros
        if all(celda == 0 for celda in estado[-1]):
            estado = estado[:-1]
            cambios = True

        # verificar si el estado esta vacio después de eliminar filas
        if not estado:
            return estado

        # elimina la primera columna si contiene solo ceros
        if all(fila[0] == 0 for fila in estado):
            estado = [fila[1:] for fila in estado]
            cambios = True

        # elimina la ultima columna si contiene solo ceros
        if estado[0] and all(fila[-1] == 0 for fila in estado):
            estado = [fila[:-1] for fila in estado]
            cambios = True

    return estado


def evolucionar(estado):
    # expande la matriz
    estado = expandir(estado)

    # genero una nueva matriz para almacenar el nuevo estado
    nuevo_estado = crear_nuevo(estado)

    for f in range(len(estado)):
        for c in range(len(estado[0])):
            n_vecinos = contar_vecinos(estado, f, c)

            # si la celula esta viva
            if estado[f][c] == 1:
                # celula muere por aislamiento o sobrepoblacion
                if n_vecinos < 2 or n_vecinos > 3:
                    nuevo_estado[f][c] = 0
                else:
                    nuevo_estado[f][c] = 1
            # si la celula esta muerta
            else:
                if n_vecinos == 3:
                    nuevo_estado[f][c] = 1
                else:
                    nuevo_estado[f][c] = 0

    return comprimir(nuevo_estado)


def contar_poblacion(estado):
    celulas = 0
    for fila in estado:
        for columna in fila:
            if columna == 1:
                celulas += 1
    return celulas


def main():
    # matriz que contiene el estado inicial del juego de la vida.
    estado = [
        [1, 1, 0],
        [0, 1, 1],
        [0, 1, 0],
    ]

    print(f"Estado inicial con poblacion de {contar_poblacion(estado)}:")
    imprimir_estado(estado)

    # iteracion para generar cada uno de los estados siguientes
    i = 0
    while True:
        # genero un nuevo estado
        estado = evolucionar(estado)

        # incremento el contado de generaciones
        i = i + 1

        # cuento la poblacion de celulas vivos
        poblacion = contar_poblacion(estado)
        print(f"Generacion {i} con poblacion de {poblacion}:")
        imprimir_estado(estado)

        # si la poblacion de celulas vivos es 0, se detiene la iteracion
        if i == 100:
            break


if __name__ == "__main__":
    main()
