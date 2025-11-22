from typing import Tuple, List, Dict, Optional
from config import *
from util import (
    h_index,
    v_index,
    check_edge_bounds,
    get_boxes_adjacent_to_edge,
    bit_is_box_complete,
    bit_is_edge_claimed
)

class DotsAndBoxesEngine:
    def __init__(self, state: Optional[DnBEngineState] = None):
        self.h_bits: int = 0
        self.v_bits: int = 0

        self.cur_player: int = 0
        if state is not None:
            self.set_state(state)

    # ---- State I/O ----
    def get_state(self) -> DnBEngineState:
        return DnBEngineState(board=(self.h_bits, self.v_bits), cur_player=self.cur_player)

    def set_state(self, state: DnBEngineState) -> None:
        """
        state = {
          "edges": [h, v],       # required
          "cur_player": 0|1,     # required
          "score": [p0, p1],     # required
        }
        검증:
          - h,v는 유효 비트 수 이내여야 함
          - score 합 == 현재 비트보드에서 완성된 박스 수
        """
        print(state)
        if not isinstance(state, DnBEngineState):
            raise TypeError("state must be a dict")

        # --- edges ---
        if (not isinstance(state.board, (list, tuple))) or len(state.board) != 2:
            raise ValueError("state.board must be [h, v]")

        h_bits, v_bits = state.board
        if not isinstance(h_bits, int) or not isinstance(v_bits, int):
            raise TypeError("board must be integers")

        h_mask = (1 << H_COUNT) - 1  # 30 bits
        v_mask = (1 << V_COUNT) - 1  # 30 bits
        # 초과 비트가 켜져 있으면 잘못된 상태
        if h_bits & ~h_mask:
            raise ValueError("H edges contain out-of-range bits")
        if h_bits & ~v_mask:
            raise ValueError("V edges contain out-of-range bits")

        # --- cur_player ---
        if state.cur_player not in (P0, P1):
            raise ValueError("state['cur_player'] must be 0 or 1")

        # 일단 반영 후 일관성 검사
        self.h_bits = h_bits
        self.v_bits = v_bits
        self.cur_player = state.cur_player

    @classmethod
    def from_state(cls, state: Dict):
        eng = cls()
        eng.set_state(state)
        return eng

    # ---- Internals ----

    def _set_edge(self, c: int, r: int, d: int) -> None:
        if d == H: self.h_bits |= (1 << h_index(c, r))
        else:      self.v_bits |= (1 << v_index(c, r))

    def _clear_edge(self, c: int, r: int, d: int) -> None:
        if d == H: self.h_bits &= ~(1 << h_index(c, r))
        else:      self.v_bits &= ~(1 << v_index(c, r))

    def is_game_over(self) -> bool:
        return (self.h_bits == (1 << H_COUNT) - 1) and \
           (self.v_bits == (1 << V_COUNT) - 1)

    # ---- API ----
    def apply_action(self, action: Action) -> Dict:
        c, r, d = action
        check_edge_bounds(action)
        if bit_is_edge_claimed(board=(self.h_bits, self.v_bits), edge=action):
            raise ValueError("Edge already claimed")

        self._set_edge(c, r, d)

        completed = [] 

        h1 = (self.h_bits >> h_index(0, 0)) & 1
        h2 = (self.h_bits >> h_index(0, 0 + 1)) & 1
        v1 = (self.v_bits >> v_index(0, 0)) & 1
        v2 = (self.v_bits >> v_index(0 + 1, 0)) & 1

        for box in get_boxes_adjacent_to_edge(action):
            if bit_is_box_complete(board=(self.h_bits, self.v_bits), box=box):
                completed.append(box)

        if completed:
            made_box = True
        else:
            made_box = False
            self.cur_player = 1 - self.cur_player

        over = self.is_game_over()

        box_mask = 0
        for (bc, br) in completed:
            box_mask |= (1 << (br * N_BOX + bc))

        return {
            "state": self.get_state(),
            "is_game_over": over,
            "is_box_completed": made_box,
            "completed_boxes": completed,
            "completed_box_mask": box_mask,
        }

    def undo_action(self,
                    action: Action,
                    player: Player) -> DnBEngineState:
        c, r, d = action
        check_edge_bounds(action)

        if not bit_is_edge_claimed((self.h_bits, self.v_bits), action):
            raise ValueError("Cannot undo: Edge is not set")

        self._clear_edge(c, r, d)

        self.cur_player = player
        over = self.is_game_over()

        return self.get_state()