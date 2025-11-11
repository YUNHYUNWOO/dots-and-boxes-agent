from typing import Tuple, List, Dict, Optional
from Util import (
    N_BOX,
    N,
    H_COUNT,
    V_COUNT,
    TOTAL_BOXES,
    H, V,
    h_index,
    v_index,
    check_bounds,
    boxes_adjacent_to_edge,
    is_box_complete,
    count_completed_boxes
)

class DotsAndBoxesEngine:
    def __init__(self, state: Optional[Dict] = None):
        self.h_bits: int = 0
        self.v_bits: int = 0
        self.cur_player: int = 0
        self.score: List[int] = [0, 0]
        if state is not None:
            self.set_state(state)

    # ---- State I/O ----
    def get_state(self) -> Dict:
        return {
            "edges": [self.h_bits, self.v_bits],
            "cur_player": self.cur_player,
            "score": self.score[:],
        }

    def set_state(self, state: Dict) -> None:
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
        if not isinstance(state, dict):
            raise TypeError("state must be a dict")

        # --- edges ---
        if "edges" not in state or not isinstance(state["edges"], (list, tuple)) or len(state["edges"]) != 2:
            raise ValueError("state['edges'] must be [h, v]")

        h, v = state["edges"]
        if not isinstance(h, int) or not isinstance(v, int):
            raise TypeError("edges must be integers")

        h_mask = (1 << H_COUNT) - 1  # 30 bits
        v_mask = (1 << V_COUNT) - 1  # 30 bits

        # 초과 비트가 켜져 있으면 잘못된 상태
        if h & ~h_mask:
            raise ValueError("H edges contain out-of-range bits")
        if v & ~v_mask:
            raise ValueError("V edges contain out-of-range bits")

        # --- cur_player ---
        if "cur_player" not in state or state["cur_player"] not in (0, 1):
            raise ValueError("state['cur_player'] must be 0 or 1")

        # --- score ---
        if "score" not in state or not isinstance(state["score"], (list, tuple)) or len(state["score"]) != 2:
            raise ValueError("state['score'] must be [p0, p1]")
        s0, s1 = state["score"]
        if not (isinstance(s0, int) and isinstance(s1, int) and s0 >= 0 and s1 >= 0):
            raise ValueError("scores must be non-negative integers")

        # 일단 반영 후 일관성 검사
        self.h_bits = h
        self.v_bits = v
        self.cur_player = state["cur_player"]
        self.score = [s0, s1]

        # 박스 개수 일관성 체크
        completed = count_completed_boxes(self.h_bits, self.v_bits)
        if s0 + s1 != completed:
            raise ValueError(
                f"Inconsistent state: sum(score)={s0+s1} but completed_boxes={completed}"
            )

    @classmethod
    def from_state(cls, state: Dict):
        eng = cls()
        eng.set_state(state)
        return eng

    # ---- Internals ----
    def _edge_is_claimed(self, c: int, r: int, d: int) -> bool:
        if d == H: return ((self.h_bits >> h_index(c, r)) & 1) == 1
        else:      return ((self.v_bits >> v_index(c, r)) & 1) == 1

    def _set_edge(self, c: int, r: int, d: int) -> None:
        if d == H: self.h_bits |= (1 << h_index(c, r))
        else:      self.v_bits |= (1 << v_index(c, r))

    def _clear_edge(self, c: int, r: int, d: int) -> None:
        if d == H: self.h_bits &= ~(1 << h_index(c, r))
        else:      self.v_bits &= ~(1 << v_index(c, r))

    def is_game_over(self) -> bool:
        return sum(self.score) == TOTAL_BOXES

    # ---- API ----
    def apply_action(self, action: Tuple[int, int, int]) -> Dict:
        c, r, d = action
        check_bounds(c, r, d)
        if self._edge_is_claimed(c, r, d):
            raise ValueError("Edge already claimed")

        self._set_edge(c, r, d)

        completed = []

        h1 = (self.h_bits >> h_index(0, 0)) & 1
        h2 = (self.h_bits >> h_index(0, 0 + 1)) & 1
        v1 = (self.v_bits >> v_index(0, 0)) & 1
        v2 = (self.v_bits >> v_index(0 + 1, 0)) & 1

        for (bc, br) in boxes_adjacent_to_edge(c, r, d):
            if is_box_complete(self.h_bits, self.v_bits, bc, br):
                completed.append((bc, br))

        if completed:
            self.score[self.cur_player] += len(completed)
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
                    action: Tuple[int, int, int],
                    completed_boxes: List[Tuple[int, int]],
                    player_turn: int) -> Dict:
        c, r, d = action
        check_bounds(c, r, d)

        if not self._edge_is_claimed(c, r, d):
            raise ValueError("Cannot undo: Edge is not set")

        self._clear_edge(c, r, d)

        if completed_boxes:
            self.score[player_turn] -= len(completed_boxes)
            if self.score[player_turn] < 0:
                raise ValueError("Undo would make score negative (invalid history)")

        self.cur_player = player_turn
        over = self.is_game_over()

        return {"state": self.get_state(), "is_game_over": over}
