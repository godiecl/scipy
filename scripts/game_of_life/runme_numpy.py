#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
from dataclasses import dataclass
from typing import ClassVar, List

import numpy as np
from typeguard import typechecked


@typechecked
@dataclass
class GameOfLife:
    """
    The Game of Life: This class implements the Game of Life based on a cellular
    automaton. It uses a numpy array to represent the state of the game.
    """

    # the state (matrix)
    state: np.ndarray

    # the current generation
    generation: int = 0

    # the max number of generations
    max_generations: int = 1000

    # the representation of a dead cell
    dead_cell_char: str = " "

    # the representatoin of a live cell
    live_cell_char: str = "࿕"

    # class constant
    ALIVE: ClassVar[int] = 1
    DEAD: ClassVar[int] = 0

    # the validation function
    def __post_init__(self) -> None:
        # the state need to be a numpy array
        if not isinstance(self.state, np.ndarray):
            raise TypeError("Game of Life must be a numpy array")

        # the state need to be a 2D numpy array
        if self.state.ndim != 2:
            raise TypeError("Game of Life must be a 2D array")

        # the state need to be a 0 (DEAD) or 1 (ALIVE)
        if not np.all(np.isin(self.state, [self.ALIVE, self.DEAD])):
            raise TypeError("Game of Life must contain only cells 0 and 1")

        # the max_generations need to be positive
        if self.max_generations <= 0:
            raise TypeError("max_iterations must be greater than 0")

    @classmethod
    @typechecked
    def from_list(cls, initial_state: List[List[int]]) -> "GameOfLife":
        """
        Create a GameOfLife object from a list of lists.
        """
        # Validate the input
        if not initial_state:
            raise ValueError("Initial state cannot be empty")

        # the input should be a list of lists
        if not all(isinstance(row, list) for row in initial_state):
            raise ValueError("Initial state must be a list of lists")

        return cls(state=np.array(initial_state))

    def population(self) -> np.int64:
        """Sum all the ALIVE cells"""
        return np.sum(self.state)

    def __str__(self) -> str:
        """Return a string representation of the current state"""
        return "\n".join(
            "".join(
                self.live_cell_char if cell == self.ALIVE else self.dead_cell_char
                for cell in row
            )
            for row in self.state
        )


def main():
    # initial state
    state = [
        [1, 1, 0],
        [0, 1, 1],
        [0, 1, 0],
    ]

    # create the object GameOfLife
    game_of_life = GameOfLife.from_list(state)

    # print the initial state
    print(f"The current live cells is: {game_of_life.population()}.")
    print(game_of_life)

if __name__ == "__main__":
    main()
