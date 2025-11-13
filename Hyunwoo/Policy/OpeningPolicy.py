from typing import List, Tuple, Dict, Optional
import numpy as np
import random
from .BasePolicy import BasePolicy
# ---------- Helpers to work with the Env observation ----------



def count_box_sides(board_lines: List[List[List[int]]], bx: int, by: int) -> int:
    """
    박스 (bx, by)의 현재 그려진 변의 개수를 센다.
    board_lines[x][y][0] : (x, y)에서 오른쪽으로 가는 가로선
    board_lines[x][y][1] : (x, y)에서 아래로 가는 세로선
    """
    # 노드 기준 크기
    C = len(board_lines)
    R = len(board_lines[0])

    # 박스 개수는 (C-1) x (R-1) 이라고 가정
    # bx in [0, C-2], by in [0, R-2]
    top    = board_lines[bx][by][0]
    bottom = board_lines[bx][by+1][0]
    left   = board_lines[bx][by][1]
    right  = board_lines[bx+1][by][1]
    return top + bottom + left + right


def boxes_touched_by_edge(x: int, y: int, d: int, W: int, H: int):
    """
    edge (x, y, d)가 인접한 박스(칸)들의 (bx, by) 리스트를 리턴.
    W = 박스 가로 개수, H = 박스 세로 개수
    """
    boxes = []
    if d == 0:  # horizontal
        # 위 박스
        if y > 0 and y - 1 < H:
            boxes.append((x, y - 1))
        # 아래 박스
        if y < H:
            boxes.append((x, y))
    else:  # d == 1 vertical
        # 왼쪽 박스
        if x > 0 and x - 1 < W:
            boxes.append((x - 1, y))
        # 오른쪽 박스
        if x < W:
            boxes.append((x, y))
    return boxes


def get_box_missing_edges(board_lines, bx, by):
    """
    박스 (bx, by)가 3변이 그려져 있다면,
    남은 1개의 변(들)을 (x, y, d) 형태로 리턴.
    (실제로는 항상 1개지만 구현상 리스트로)
    """
    missing = []
    # 위, 아래, 왼, 오
    if board_lines[bx][by][0] == 0:       # top
        missing.append((bx, by, 0))
    if board_lines[bx][by+1][0] == 0:     # bottom
        missing.append((bx, by+1, 0))
    if board_lines[bx][by][1] == 0:       # left
        missing.append((bx, by, 1))
    if board_lines[bx+1][by][1] == 0:     # right
        missing.append((bx+1, by, 1))
    return missing


def dots_and_boxes_policy(edges: List[List[List[int]]]):
    """
    board_lines[c][r][d] 입력으로부터 한 수를 선택해서 (x, y, d)를 리턴.
    
    정책:
    1) 3변이 이미 그려진 박스가 있으면 → 그 박스를 완성하는 선 중 하나를 선택.
    2) 없으면 → 안전수(어떤 박스도 3변이 되지 않는 수) 중에서 랜덤.
    3) 안전수도 없으면 → 남은 아무 수나 랜덤.
    """
    C = len(edges)
    R = len(edges[0])
    D = len(edges[0][0])
    assert D == 2, "이 구현은 d=0(h), d=1(v) 두 방향만 가정함."

    W = C - 1  # 박스 가로 개수
    H = R - 1  # 박스 세로 개수

    available_moves = []

    # 1. 모든 아직 안 그려진 선(엣지) 수집
    for x in range(C):
        for y in range(R):
            for d in range(D):
                # 실제로 존재하는 선만 고려 (범위 밖은 패스)
                if d == 0:
                    # horizontal: x in [0, W-1], y in [0, H]
                    if not (0 <= x < W and 0 <= y <= H):
                        continue
                else:
                    # vertical: x in [0, W], y in [0, H-1]
                    if not (0 <= x <= W and 0 <= y < H):
                        continue

                if edges[x][y][d] == 0:  # 아직 안 그려진 선만
                    available_moves.append((x, y, d))

    if not available_moves:
        return None  # 둘 곳이 없음 (게임 종료 상태)

    # 2. 먼저, 3변 박스를 찾아서 완성시킬 수 있는 수들 찾기
    complete_box_moves = []
    for bx in range(W):
        for by in range(H):
            sides = count_box_sides(edges, bx, by)
            if sides == 3:
                missing = get_box_missing_edges(edges, bx, by)
                # missing 중 실제로 available_moves에 있는 것만 사용
                for mv in missing:
                    if mv in available_moves:
                        complete_box_moves.append(mv)

    if complete_box_moves:
        return random.choice(complete_box_moves)

    # 3. 안전수(safe moves) 찾기:
    #    이 수를 두었을 때, 인접한 박스들 중 어느 것도 3변이 되지 않으면 safe.
    safe_moves = []

    for (x, y, d) in available_moves:
        boxes = boxes_touched_by_edge(x, y, d, W, H)
        unsafe = False
        for (bx, by) in boxes:
            sides_before = count_box_sides(edges, bx, by)
            sides_after = sides_before + 1
            if sides_after == 3:
                unsafe = True
                break
        if not unsafe:
            safe_moves.append((x, y, d))

    if safe_moves:
        return random.choice(safe_moves)

    # 4. 안전수도 없으면 그냥 남은 수 중 랜덤
    return random.choice(available_moves)

class OpeningPolicy(BasePolicy):

    def get_action(self, observation: Dict[str, np.ndarray], info: Dict, env) -> Tuple[int,int,int]:
        
        return list(dots_and_boxes_policy(observation['edges']))
