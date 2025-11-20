import random

from config import *
from util import (count_box_edge, get_boxes_adjacent_to_edge, get_box_missing_edges)


from .basepolicy import BasePolicy, TimeManager
# ---------- Helpers to work with the Env observation ----------

def dots_and_boxes_policy(board: Board) -> Action:
    """
    board_lines[c][r][d] 입력으로부터 한 수를 선택해서 (x, y, d)를 리턴.
    
    정책:
    1) 3변이 이미 그려진 박스가 있으면 → 그 박스를 완성하는 선 중 하나를 선택.
    2) 없으면 → 안전수(어떤 박스도 3변이 되지 않는 수) 중에서 랜덤.
    3) 안전수도 없으면 → 남은 아무 수나 랜덤.
    """
    available_moves = []

    # 1. 모든 아직 안 그려진 선(엣지) 수집
    for c in range(N):
        for r in range(N):
            for d in range(2):
                # 실제로 존재하는 선만 고려 (범위 밖은 패스)
                if d == 0:
                    # horizontal: x in [0, W-1], y in [0, H]
                    if not (0 <= c < N_BOX and 0 <= r <= N_BOX):
                        continue
                else:
                    # vertical: x in [0, W], y in [0, H-1]
                    if not (0 <= c <= N_BOX and 0 <= r < N_BOX):
                        continue

                if board[c][r][d] == 0:  # 아직 안 그려진 선만
                    available_moves.append((c, r, d))

    if not available_moves:
        return None  # 둘 곳이 없음 (게임 종료 상태)

    # 2. 먼저, 3변 박스를 찾아서 완성시킬 수 있는 수들 찾기
    complete_box_moves = []
    for br in range(N_BOX):
        for bc in range(N_BOX):
            box = (br, bc)
            sides = count_box_edge(board, box)
            if sides == 3:
                missing = get_box_missing_edges(board, box)
                # missing 중 실제로 available_moves에 있는 것만 사용
                for mv in missing:
                    if mv in available_moves:
                        complete_box_moves.append(mv)

    if complete_box_moves:
        return random.choice(complete_box_moves)

    # 3. 안전수(safe moves) 찾기:
    #    이 수를 두었을 때, 인접한 박스들 중 어느 것도 3변이 되지 않으면 safe.
    safe_moves = []

    for action in available_moves:
        boxes = get_boxes_adjacent_to_edge(action)
        unsafe = False
        for box in boxes:
            sides_before = count_box_edge(board, box)
            sides_after = sides_before + 1
            if sides_after == 3:
                unsafe = True
                break
        if not unsafe:
            safe_moves.append(action)

    if safe_moves:
        return random.choice(safe_moves)

    # 4. 안전수도 없으면 그냥 남은 수 중 랜덤
    return random.choice(available_moves)

class OpeningPolicy(BasePolicy):
    def get_action(self, observation: dict, time_manager:TimeManager) -> Action:
        return list(dots_and_boxes_policy(observation['board']))
