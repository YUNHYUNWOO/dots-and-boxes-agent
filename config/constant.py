from typing import Any, Callable, Iterable, Optional, NamedTuple, List, Literal
from dataclasses import dataclass

N_BOX = 5
N = N_BOX + 1
H_COUNT = N * (N - 1)   # 30
V_COUNT = (N - 1) * N   # 30
TOTAL_BOXES = N_BOX * N_BOX  # 25
H, V = 0, 1
Dir = Literal[0, 1]

TIME_LIMIT = 24.0

P0, P1 = 0, 1
Player = Literal[0, 1]
Action = tuple[int, int, Dir] # (c, r, d)
Edge = tuple[int, int, Dir] # (c, r, d)
Box = tuple[int, int] # (c, r, d)

Board = list[list[list[int]]]
BitBoard = tuple[int, int]

@dataclass
class DnBEngineState():
    board: BitBoard
    cur_player: Player