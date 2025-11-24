from config import *

def get_boxes_adjacent_to_edge(edge:Edge) -> list[Box]:
    c, r, d = edge

    boxes = []
    if d == H:
        if 0 <= r - 1 < N_BOX: boxes.append((c, r - 1))
        if 0 <= r < N_BOX:     boxes.append((c, r))
    else:
        if 0 <= c - 1 < N_BOX: boxes.append((c - 1, r))
        if 0 <= r < N_BOX and 0 <= c < N_BOX: boxes.append((c, r))
    return boxes

def get_edges_adjacent_to_box(box: Box) -> list[Edge]:
    c, r = box
    return [
        (c, r, 0),
        (c + 1, r, 1),
        (c, r + 1, 0),
        (c, r, 1)
    ]

def count_box_edges(board: Board, box : Box)-> int:
    """
    Count all edges of box from given board and box
    0 <= bc < N_BOX, 0 <= br < N_BOX
    board[c][r][0] : (x, y)에서 오른쪽으로 가는 가로선
    board[c][r][1] : (x, y)에서 아래로 가는 세로선
    """

    bc, br = box

    top    = board[bc][br][0]
    bottom = board[bc][br+1][0]
    left   = board[bc][br][1]
    right  = board[bc+1][br][1]

    return top + bottom + left + right

def get_missing_edges(board: Board, box: Box) -> list[Action]:
    """
    get list of missing edges of given box
    board: Board[list of shape [c, r, d]]
    Box: Box[bc, br]
    """
    bc, br = box

    missing = []
    # 위, 아래, 왼, 오
    if board[bc][br][0] == 0:       # top
        missing.append((bc, br, 0))
    if board[bc][br+1][0] == 0:     # bottom
        missing.append((bc, br+1, 0))
    if board[bc][br][1] == 0:       # left
        missing.append((bc, br, 1))
    if board[bc+1][br][1] == 0:     # right
        missing.append((bc+1, br, 1))
        
    return missing
