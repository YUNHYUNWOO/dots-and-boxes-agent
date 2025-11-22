from config import *

def check_edge_bounds(edge:Edge|Action) -> None:
    c, r, d = edge
    if not (0 <= r < N and 0 <= c < N and d in (H, V)):
        raise ValueError("Action out of bounds: r,c in [0,5], d in {0,1}")
    if d == H and c >= N - 1:
        raise ValueError("Invalid H edge: c must be <= 4")
    if d == V and r >= N - 1:
        raise ValueError("Invalid V edge: r must be <= 4")
    

def check_box_bounds(box: Box) -> None:
    bc, br = box
    if not 0 <= bc < N_BOX:
        raise ValueError('bc must be in [0, N_BOX-1]')
    if not 0 <= br < N_BOX:
        raise ValueError('br must be in [0, N_BOX-1]')

def check_board_bounds(board: Board) -> None:
    """
    Board = list[list[list[int]]]  형태의 3D 구조가
    DotsAndBoxes 엔진이 기대하는 크기와 일치하는지 검사한다.

    기대 형태:
        board[c][r][d]
            c: 0 ~ N-1
            r: 0 ~ N-1
            d: 0 or 1  (H or V)
    즉, shape = (N, N, 2)
    """

    # 1. 1차원 길이 (columns)
    C = len(board)
    if C != N:
        raise ValueError(f"Invalid board C dimension: got {C}, expected {N}")

    # 2. 2차원 길이 (rows)
    R = len(board[0])
    if R != N:
        raise ValueError(f"Invalid board R dimension: got {R}, expected {N}")

    # 3. 3차원 길이 (direction)
    D = len(board[0][0])
    if D != 2:
        raise ValueError(f"Invalid board D dimension: got {D}, expected 2")

    # 4. 내부 구조가 모두 같은 shape을 가지는지 검증 (optional but robust)
    for c in range(C):
        if len(board[c]) != N:
            raise ValueError(f"Row length mismatch at c={c}: expected {N}")

        for r in range(R):
            if len(board[c][r]) != 2:
                raise ValueError(
                    f"Direction length mismatch at (c={c}, r={r}): expected length 2"
                )
