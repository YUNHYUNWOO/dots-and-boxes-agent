"""Validation helpers for Board, Edge, and Box coordinates."""

from config import H, N, N_BOX, V, Action, Board, Box, Edge


def check_edge_bounds(edge: Action | Edge) -> None:
    """Validate that an edge/action uses in-bounds (c, r, d)."""

    c, r, d = edge
    if not (0 <= r < N and 0 <= c < N and d in (H, V)):
        raise ValueError("Action out of bounds: r,c in [0,5], d in {0,1}")
    if d == H and c >= N - 1:
        raise ValueError("Invalid H edge: c must be <= 4")
    if d == V and r >= N - 1:
        raise ValueError("Invalid V edge: r must be <= 4")


def check_box_bounds(box: Box) -> None:
    """Validate that a box index is inside the playable grid."""

    bc, br = box
    if not 0 <= bc < N_BOX:
        raise ValueError("bc must be in [0, N_BOX-1]")
    if not 0 <= br < N_BOX:
        raise ValueError("br must be in [0, N_BOX-1]")


def check_board_bounds(board: Board) -> None:
    """Validate a Board has shape (N, N, 2) with consistent inner lengths."""

    cols = len(board)
    if cols != N:
        raise ValueError(f"Invalid board C dimension: got {cols}, expected {N}")

    rows = len(board[0])
    if rows != N:
        raise ValueError(f"Invalid board R dimension: got {rows}, expected {N}")

    dirs = len(board[0][0])
    if dirs != 2:
        raise ValueError(f"Invalid board D dimension: got {dirs}, expected 2")

    for c in range(cols):
        if len(board[c]) != N:
            raise ValueError(f"Row length mismatch at c={c}: expected {N}")

        for r in range(rows):
            if len(board[c][r]) != 2:
                raise ValueError(
                    f"Direction length mismatch at (c={c}, r={r}): expected length 2"
                )
