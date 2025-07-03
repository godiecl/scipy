#  Copyright (c) 2025. Departamento de Ingenieria de Sistemas y Computacion.
import logging
from dataclasses import dataclass
from typing import ClassVar, List, Optional

import numpy as np
import seaborn as sns
from benchmark import benchmark
from logger import configure_logging
from matplotlib import pyplot as plt
from scipy.ndimage import convolve
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

    # kernel to count the neighbors
    NEIGHBOR_KERNEL = np.array(
        [
            [1, 1, 1],  #
            [1, 0, 1],  #
            [1, 1, 1],  #
        ],
        dtype=np.uint8,
    )

    @typechecked
    def __post_init__(self) -> None:
        """Post-initialization checks for the Game of Life class."""

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

    # @typechecked
    def population(self) -> np.uint64:
        """sum all the ALIVE cells."""
        return np.sum(self.state, dtype=np.uint64)

    @typechecked
    def __str__(self) -> str:
        """Return a string representation of the current state."""
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

    # @typechecked
    # noinspection PyMethodMayBeStatic
    def _compress(self, state: np.ndarray) -> np.ndarray:
        """Compress the state by removing border rows and columns that contain only dead cells (zeros)."""

        if state.size == 0:
            return np.zeros((1, 1), dtype=state.dtype)

        # find rows and columns with any live cells
        live_rows = np.any(state != 0, axis=1)
        live_cols = np.any(state != 0, axis=0)

        if not live_rows.any() or not live_cols.any():
            return np.zeros((1, 1), dtype=state.dtype)

        # get bounding box indices
        row_min, row_max = np.where(live_rows)[0][[0, -1]]
        col_min, col_max = np.where(live_cols)[0][[0, -1]]

        return state[row_min : row_max + 1, col_min : col_max + 1]

    # @typechecked
    def evolve(self) -> "GameOfLife":
        """Evolve the current state to the next generation."""

        # padded state around the original state
        padded_state = np.pad(
            self.state,
            pad_width=1,
            mode="constant",
            constant_values=self.DEAD,  # fill with dead cells
        )

        # count the neighbors using convolution
        neighbors = convolve(
            padded_state,
            self.NEIGHBOR_KERNEL,
            mode="constant",
            cval=self.DEAD,  # fill with dead cells
        )

        # determine which cells are alive
        alive = padded_state == self.ALIVE

        # apply the rules of the Game of Life vectorized
        # https://numpy.org/doc/stable/reference/generated/numpy.where.html
        new_state = np.where(
            (alive & ((neighbors == 2) | (neighbors == 3)))
            | (~alive & (neighbors == 3)),
            self.ALIVE,
            self.DEAD,
        ).astype(np.uint8)

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

        # iterate over the number of generations
        for _ in tqdm(
            range(0, max_generations),
            desc="Evolving generations",
            unit="gen",
            ncols=200,  # progress bar width
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
    max_generations = 3000
    log.debug(f"Running simulation for {max_generations} generations ...")
    with benchmark(operation_name="run_simulation", log=log):
        game_of_life.run_simulation(max_generations, show_progress=True)

    # print the final state
    # print(game_of_life)
    # plot_game_of_life(game_of_life)
    # plot_game_of_life(game_of_life, "../../output/")

    log.debug("Done.")


if __name__ == "__main__":
    # Run the main in a profile fashion
    # cProfile.run("main()", "../../output/game_of_life.prof")
    main()
