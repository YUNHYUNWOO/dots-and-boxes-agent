"""Utility helpers for Board-level operations using (c, r, d) coordinates."""

from config import H, N_BOX, Action, Board, Box, Edge


def get_boxes_adjacent_to_edge(edge: Edge) -> list[Box]:
    """Return boxes that share the given edge."""

    c, r, d = edge

    boxes: list[Box] = []
    if d == H:
        if 0 <= r - 1 < N_BOX:
            boxes.append((c, r - 1))
        if 0 <= r < N_BOX:
            boxes.append((c, r))
    else:
        if 0 <= c - 1 < N_BOX:
            boxes.append((c - 1, r))
        if 0 <= r < N_BOX and 0 <= c < N_BOX:
            boxes.append((c, r))
    return boxes


def get_edges_adjacent_to_box(box: Box) -> list[Edge]:
    """Return the four edges surrounding a box (top, right, bottom, left)."""

    c, r = box
    return [
        (c, r, H),
        (c + 1, r, 1),
        (c, r + 1, H),
        (c, r, 1),
    ]


def count_box_edges(board: Board, box: Box) -> int:
    """Count how many of the box's four edges are present on the Board."""

    bc, br = box

    top = board[bc][br][0]
    bottom = board[bc][br + 1][0]
    left = board[bc][br][1]
    right = board[bc + 1][br][1]

    return top + bottom + left + right


def get_missing_edges(board: Board, box: Box) -> list[Action]:
    """Return the list of missing edges for a given box."""

    bc, br = box

    missing: list[Action] = []
    if board[bc][br][0] == 0:
        missing.append((bc, br, 0))
    if board[bc][br + 1][0] == 0:
        missing.append((bc, br + 1, 0))
    if board[bc][br][1] == 0:
        missing.append((bc, br, 1))
    if board[bc + 1][br][1] == 0:
        missing.append((bc + 1, br, 1))

    return missing
