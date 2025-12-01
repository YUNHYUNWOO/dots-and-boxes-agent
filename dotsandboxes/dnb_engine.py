"""Bitboard-based engine for Dots and Boxes using (c, r, d) coordinates."""

from typing import Dict, Optional

from config import (
    Action,
    BitBoard,
    DnBEngineState,
    H,
    H_COUNT,
    N_BOX,
    P0,
    P1,
    Player,
    V,
    V_COUNT,
)
from util.bit_dnb_util import bit_is_box_complete, bit_is_edge_claimed
from util.dnb_util import get_boxes_adjacent_to_edge
from util.validate import check_edge_bounds
from util.bit_dnb_util import h_index, v_index


class DotsAndBoxesEngine:
    """Game engine operating on BitBoard representations."""

    def __init__(self, state: Optional[DnBEngineState] = None):
        self.h_bits: int = 0
        self.v_bits: int = 0
        self.cur_player: int = 0
        if state is not None:
            self.set_state(state)

    # ---- State I/O ----
    def get_state(self) -> DnBEngineState:
        """Return a snapshot of the current engine state."""

        return DnBEngineState(board=(self.h_bits, self.v_bits), cur_player=self.cur_player)

    def set_state(self, state: DnBEngineState) -> None:
        """Validate and apply an incoming engine state."""

        if not isinstance(state, DnBEngineState):
            raise TypeError("state must be a DnBEngineState")

        if not isinstance(state.board, (list, tuple)) or len(state.board) != 2:
            raise ValueError("state.board must be (h_bits, v_bits)")

        h_bits, v_bits = state.board
        if not isinstance(h_bits, int) or not isinstance(v_bits, int):
            raise TypeError("board must contain integers")

        h_mask = (1 << H_COUNT) - 1
        v_mask = (1 << V_COUNT) - 1
        if h_bits & ~h_mask:
            raise ValueError("H edges contain out-of-range bits")
        if v_bits & ~v_mask:
            raise ValueError("V edges contain out-of-range bits")

        if state.cur_player not in (P0, P1):
            raise ValueError("state.cur_player must be 0 or 1")

        self.h_bits = h_bits
        self.v_bits = v_bits
        self.cur_player = state.cur_player

    @classmethod
    def from_state(cls, state: Dict) -> "DotsAndBoxesEngine":
        """Alternate constructor from a dict-like state."""

        eng = cls()
        eng.set_state(state)
        return eng

    # ---- Internals ----
    def _set_edge(self, c: int, r: int, d: int) -> None:
        if d == H:
            self.h_bits |= 1 << h_index(c, r)
        else:
            self.v_bits |= 1 << v_index(c, r)

    def _clear_edge(self, c: int, r: int, d: int) -> None:
        if d == H:
            self.h_bits &= ~(1 << h_index(c, r))
        else:
            self.v_bits &= ~(1 << v_index(c, r))

    def is_game_over(self) -> bool:
        """Return True when all edges are claimed."""

        return (self.h_bits == (1 << H_COUNT) - 1) and (self.v_bits == (1 << V_COUNT) - 1)

    # ---- API ----
    def apply_action(self, action: Action) -> Dict:
        """Apply an action and return metadata about the move."""

        c, r, d = action
        check_edge_bounds(action)
        if bit_is_edge_claimed(board=(self.h_bits, self.v_bits), edge=action):
            raise ValueError("Edge already claimed")

        self._set_edge(c, r, d)

        completed = []
        for box in get_boxes_adjacent_to_edge(action):
            if bit_is_box_complete(board=(self.h_bits, self.v_bits), box=box):
                completed.append(box)

        made_box = bool(completed)
        if not made_box:
            self.cur_player = 1 - self.cur_player

        over = self.is_game_over()

        box_mask = 0
        for (bc, br) in completed:
            box_mask |= 1 << (br * N_BOX + bc)

        return {
            "state": self.get_state(),
            "is_game_over": over,
            "is_box_completed": made_box,
            "completed_boxes": completed,
            "completed_box_mask": box_mask,
        }

    def undo_action(self, action: Action, player: Player) -> DnBEngineState:
        """Undo an action and restore player turn."""

        c, r, d = action
        check_edge_bounds(action)

        if not bit_is_edge_claimed((self.h_bits, self.v_bits), action):
            raise ValueError("Cannot undo: edge is not set")

        self._clear_edge(c, r, d)
        self.cur_player = player

        return self.get_state()
