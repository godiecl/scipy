#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging
from dataclasses import dataclass
from typing import ClassVar, List, Optional

import numpy as np
import seaborn as sns
from benchmark import benchmark
from logger import configure_logging
from matplotlib import pyplot as plt
from tqdm import tqdm
from typeguard import typechecked


@dataclass
class GameOfLife:
    """
    The Conway's Game of Life: This class implements the game on a cellular automaton.
    It uses a numpy array to represent the state of the game.
    """

    # the state (matrix)
    state: np.ndarray

    # the current generation
    generation: int = np.int64(0)

    # the max number of generations
    max_generations: int = np.int64(1000)

    # the representation of a dead cell
    dead_cell_char: str = " "

    # the representatoin of a live cell
    live_cell_char: str = "࿕"

    # class constant
    ALIVE: ClassVar[int] = 1
    DEAD: ClassVar[int] = 0

    # the validation function
    @typechecked
    def __post_init__(self) -> None:
        # the state need to be a numpy array
        if not isinstance(self.state, np.ndarray):
            raise TypeError("Game of Life must be a numpy array")

        # convert the state to a numpy array of uint8
        if self.state.dtype != np.uint8:
            self.state = self.state.astype(np.uint8)

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
        """Create a GameOfLife object from a list of lists."""

        # validate the input
        if not initial_state:
            raise ValueError("Initial state cannot be empty")

        # the input should be a list of lists
        if not all(isinstance(row, list) for row in initial_state):
            raise ValueError("Initial state must be a list of lists")

        return cls(state=np.array(initial_state, dtype=np.uint8))

    @typechecked
    def population(self) -> np.uint64:
        """sum all the ALIVE cells"""
        return np.sum(self.state, dtype=np.uint64)

    @typechecked
    def __str__(self) -> str:
        """
        Return a string representation of the current state
        """
        grid = "\n".join(
            "".join(
                self.live_cell_char if cell == self.ALIVE else self.dead_cell_char
                for cell in row
            )
            for row in self.state
        )
        header = (
            f"Generation: {self.generation:04d} | Population: {self.population():04d}\n"
        )
        separator = "-" * max(len(line) for line in grid.split("\n")) + "\n"
        return header + separator + grid

    @typechecked
    def _count_neighbors(self, state: np.ndarray, row: int, col: int) -> np.uint8:
        """
        Count the number of alive neighbors for a given cell in the state.
        """
        neighbors = 0

        for r in range(row - 1, row + 2):
            for c in range(col - 1, col + 2):
                # skip the cell itself
                if r == row and c == col:
                    continue
                # check if the neighbor is within bounds
                if 0 <= r < state.shape[0] and 0 <= c < state.shape[1]:
                    neighbors += state[r, c]
        return np.uint8(neighbors)

    @typechecked
    def _compress(self, state: np.ndarray) -> np.ndarray:
        """Compress the state by removing border rows and columns that contain only dead cells (zeros)."""
        # handle edge case: if array is empty or has zero dimensions
        if state.size == 0 or state.shape[0] == 0 or state.shape[1] == 0:
            return np.zeros((1, 1), dtype=state.dtype)

        # find rows and columns that have at least one live cell (non-zero)
        live_rows = np.any(state != 0, axis=1)
        live_cols = np.any(state != 0, axis=0)

        # if no live cells exist, return a minimal grid
        if not np.any(live_rows) or not np.any(live_cols):
            return np.zeros((1, 1), dtype=state.dtype)

        # find the boundaries of live cells
        row_indices = np.where(live_rows)[0]
        col_indices = np.where(live_cols)[0]

        # extract the minimal bounding box containing all live cells
        min_row, max_row = row_indices[0], row_indices[-1]
        min_col, max_col = col_indices[0], col_indices[-1]

        return state[min_row : max_row + 1, min_col : max_col + 1]

    @typechecked
    def evolve(self) -> "GameOfLife":
        """Evolve the current state to the next generation."""

        # the dimensions of the state
        rows, cols = self.state.shape

        # expand the state: one row and column at the beginning and end
        expanded_state = np.zeros((rows + 2, cols + 2))

        # fill the expanded state with the current state in the middle
        expanded_state[1:-1, 1:-1] = self.state

        # the new state is the same size as the expanded state
        new_state = np.zeros(expanded_state.shape)

        # iterate over the expanded state
        for r in range(rows + 2):
            for c in range(cols + 2):
                # count the number of alive neighbors
                neighbors = self._count_neighbors(expanded_state, r, c)

                # apply the rules of the game
                if expanded_state[r, c] == self.ALIVE:
                    if neighbors < 2 or neighbors > 3:
                        new_state[r, c] = self.DEAD
                    else:
                        new_state[r, c] = self.ALIVE
                else:
                    if neighbors == 3:
                        new_state[r, c] = self.ALIVE

        # assign the new_state pos-compression
        self.state = self._compress(new_state)

        # increment the generation
        self.generation += 1
        return self

    @typechecked
    def run_simulation(
        self,
        max_generations: Optional[int] = None,
        show_progress: Optional[bool] = False,
    ) -> str:
        """Run the simulation for a given number of generations."""

        if max_generations is None:
            max_generations = self.max_generations

        if max_generations <= 0:
            raise ValueError("max_generations must be positive")

        for _ in tqdm(
            range(0, max_generations),
            desc="Evolving generations",
            unit="gen",
            ncols=200,
            disable=not show_progress,
        ):
            # generate the next generation
            self.evolve()
            # if the population is 0, break the loop
            if self.population() == 0:
                return "WARN: Stopping simulation at:\n" + str(self)

        return str(self)


@typechecked
def plot_game_of_life(game_of_life: GameOfLife, path: Optional[str] = None) -> None:
    """Draw the Game of Life using matplotlib and pyplot."""

    # define the figure
    fig = plt.figure(facecolor="white", dpi=200)

    # get the current axis
    ax = plt.gca()

    # hide the axis
    ax.set_axis_off()

    sns.heatmap(
        game_of_life.state,  # ndarray
        cmap="binary",
        cbar=False,
        square=True,
        linewidths=0.25,
        linecolor="#f0f0f0",  # rgb
        ax=ax,
    )

    # set the title
    plt.title("The Conway's Game of Life")

    # create some stats
    # the total of space inside the grid
    total_space = game_of_life.state.shape[0] * game_of_life.state.shape[1]
    density = game_of_life.population() / total_space

    stats = (
        f"Generation: {game_of_life.generation}\n"
        f"Population: {game_of_life.population()}\n"
        f"Grid size: {game_of_life.state.shape[0]} x {game_of_life.state.shape[1]}\n"
        f"Density: {density:.2f}"
    )

    # plot the stats
    plt.figtext(
        0.99,
        0.01,
        stats,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8, boxstyle="round", pad=0.5),
    )

    # compress the layout
    plt.tight_layout()

    # we need to save the plot?
    if path is not None:
        plt.savefig(
            f"{path}/game_of_life-{game_of_life.generation:04d}.png",
            dpi=200,
            bbox_inches="tight",
        )

    # show time !
    plt.show()


def main():
    # configure the logger
    configure_logging(logging.DEBUG)

    # get the logger
    log = logging.getLogger(__name__)
    log.debug("Starting main ..")

    # initial state
    state = [
        [1, 1, 0],
        [0, 1, 1],
        [0, 1, 0],
    ]

    # create the object GameOfLife
    game_of_life = GameOfLife.from_list(state)

    # print the initial state
    print(game_of_life)

    # run the simulation
    with benchmark(operation_name="run_simulation", log=log):
        game_of_life.run_simulation(max_generations=1000, show_progress=True)

    # print the final state
    print(game_of_life)
    # plot_game_of_life(game_of_life)
    # plot_game_of_life(game_of_life, "../../output/")

    log.debug("Done.")


if __name__ == "__main__":
    main()
