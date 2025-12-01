"""Shared constants and type aliases for the Dots and Boxes environment.

Coordinate convention:
    (c, r, d) stands for column, row, and direction where H=0, V=1.
Board encodings:
    - ``Board``: nested lists marking edge occupancy.
    - ``BitBoard``: two bitmasks (horizontal, vertical) to store edges compactly.
"""

from dataclasses import dataclass
from typing import Literal

# Board dimensions
N_BOX = 5  # Number of boxes along one side of the board (i.e., a 5x5 grid)
N = N_BOX + 1  # Number of grid points (corners) along one side
H_COUNT = N * (N - 1)  # Total number of horizontal edges
V_COUNT = (N - 1) * N  # Total number of vertical edges
TOTAL_BOXES = N_BOX * N_BOX  # Total number of boxes on the board
H, V = 0, 1
Dir = Literal[0, 1]

# Game configuration
TIME_LIMIT = 24.0  # Default total time budget (seconds) allocated to each player

# Player identifiers
P0, P1 = 0, 1
Player = Literal[0, 1]
Action = tuple[int, int, Dir]  # (column, row, direction)
Edge = tuple[int, int, Dir]  # (column, row, direction)
Box = tuple[int, int]  # (column, row)

# Board encodings
Board = list[list[list[int]]]
BitBoard = tuple[int, int]


@dataclass
class DnBEngineState:
    """Lightweight snapshot of the engine state for search and simulation."""

    board: BitBoard
    cur_player: Player
