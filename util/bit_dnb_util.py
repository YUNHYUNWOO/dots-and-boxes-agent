"""Bitboard helpers for Dots and Boxes.

All coordinates follow the (c, r, d) convention: column, row, direction
with H=0 (horizontal) and V=1 (vertical). BitBoard stores two bitmasks
for horizontal and vertical edges respectively, distinct from the nested
list `Board` representation.
"""

import random
import time
from typing import List

import pandas

from config import H, N, N_BOX, V, Action, BitBoard, Board, Box, Edge
from util.dnb_util import get_boxes_adjacent_to_edge


def h_index(c: int, r: int) -> int:
    """Flatten (c, r) into a bit index for horizontal edges."""

    return r * (N - 1) + c


def v_index(c: int, r: int) -> int:
    """Flatten (c, r) into a bit index for vertical edges."""

    return r * N + c

def bit_count_edges(board: BitBoard) -> int:
    h_bits, v_bits = board
    return h_bits.bit_count() + v_bits.bit_count()

def bit_is_box_complete(board: BitBoard, box: Box) -> bool:
    """Return True if all four edges of ``box`` are claimed."""

    bc, br = box
    h_bits, v_bits = board

    h1 = (h_bits >> h_index(bc, br)) & 1
    h2 = (h_bits >> h_index(bc, br + 1)) & 1
    v1 = (v_bits >> v_index(bc, br)) & 1
    v2 = (v_bits >> v_index(bc + 1, br)) & 1

    return (h1 & h2 & v1 & v2) == 1


def bit_count_completed_boxes(board: BitBoard) -> int:
    """Count how many boxes are completed on the bitboard."""

    cnt = 0
    for br in range(N_BOX):
        for bc in range(N_BOX):
            if bit_is_box_complete(board, box=(bc, br)):
                cnt += 1
    return cnt


def bit_is_edge_claimed(board: BitBoard, edge: Edge) -> bool:
    """Return True if the given edge is already claimed on the bitboard."""

    c, r, d = edge
    if d == H:
        return ((board[0] >> h_index(c, r)) & 1) == 1
    return ((board[1] >> v_index(c, r)) & 1) == 1


def encode_board(board: Board) -> BitBoard:
    """Convert a Board into a compact BitBoard (h_bits, v_bits)."""

    h_bits, v_bits = 0, 0
    for r in range(N):
        for c in range(N):
            if c != N - 1 and board[c][r][H]:
                h_bits |= 1 << h_index(c, r)
            if r != N - 1 and board[c][r][V]:
                v_bits |= 1 << v_index(c, r)

    return h_bits, v_bits


def decode_bitboard(bit_board: BitBoard) -> Board:
    """Convert a BitBoard back into the list-based Board representation."""

    h_bits, v_bits = bit_board
    board: Board = [[[0 for _ in range(2)] for _ in range(N)] for _ in range(N)]

    for r in range(N):
        for c in range(N):
            if c != N - 1:
                board[c][r][H] = ((h_bits >> h_index(c, r)) & 1) == 1
            if r != N - 1:
                board[c][r][V] = ((v_bits >> v_index(c, r)) & 1) == 1

    return board


def bit_get_legal_actions(bit_board: BitBoard) -> List[Action]:
    """Return all unclaimed edges encoded as (c, r, d) tuples."""

    h_bits, v_bits = bit_board

    h_bits &= (1 << (N * (N - 1))) - 1
    v_bits &= (1 << ((N - 1) * N)) - 1

    actions: List[Action] = []

    # Horizontal edges: r in [0..N-1], c in [0..N-2]
    for r in range(N):
        row_bits = (h_bits >> (r * (N - 1))) & ((1 << (N - 1)) - 1)
        for c in range(N - 1):
            if ((row_bits >> c) & 1) == 0:
                actions.append((c, r, H))

    # Vertical edges: r in [0..N-2], c in [0..N-1]
    for r in range(N - 1):
        row_bits = (v_bits >> (r * N)) & ((1 << N) - 1)
        for c in range(N):
            if ((row_bits >> c) & 1) == 0:
                actions.append((c, r, V))

    return actions


def bit_count_box_edges(board: BitBoard, box: Box) -> int:
    """Count how many edges of ``box`` are already claimed on the bitboard."""

    h_bits, v_bits = board
    bc, br = box

    cnt = 0
    if (h_bits >> h_index(bc, br)) & 1:
        cnt += 1
    if (h_bits >> h_index(bc, br + 1)) & 1:
        cnt += 1
    if (v_bits >> v_index(bc, br)) & 1:
        cnt += 1
    if (v_bits >> v_index(bc + 1, br)) & 1:
        cnt += 1
    return cnt


def bit_makes_third_edge(board: BitBoard, action: Action) -> bool:
    """Return True if placing ``action`` creates a third edge on any adjacent box."""

    for box in get_boxes_adjacent_to_edge(action):
        if bit_count_box_edges(board=board, box=box) == 2:
            return True
    return False


def _test_get_available_actions_encoded() -> None:
    for _ in range(10_000):
        h_bits = random.randint(0, 1 << 30)
        v_bits = random.randint(0, 1 << 30)

        encoded_edges = (h_bits, v_bits)
        actions = bit_get_legal_actions(bit_board=encoded_edges)

        actions_edges = [[[0 for _ in range(2)] for _ in range(6)] for _ in range(6)]
        for action in actions:
            c, r, z = action
            actions_edges[c][r][z] = 1

        decoded_edges = decode_bitboard(encoded_edges)

        for z in range(2):
            for r in range(6):
                for c in range(6):
                    print(decoded_edges[c][r][z], end=" ")
                print()
            print("=====")

        h_comp, v_comp = encode_board(actions_edges)
        assert (h_comp & h_bits) == 0 and (v_comp & v_bits) == 0


def _test_encode_decode() -> None:
    durations = []

    for _ in range(100):
        start = time.time()

        h_bits = random.randint(0, 1 << 30)
        v_bits = random.randint(0, 1 << 30)

        edges = decode_bitboard((h_bits, v_bits))
        h_enc, v_enc = encode_board(edges)

        assert h_enc == h_bits and v_enc == v_bits
        end = time.time()
        durations.append(end - start)

    t_sr = pandas.Series(durations)
    print(t_sr.head())
    print(t_sr.mean(), t_sr.std())


if __name__ == "__main__":
    edges: Board = [[[0 for _ in range(2)] for _ in range(6)] for _ in range(6)]
    edges[3][1][0] = 1
    print(decode_bitboard(encode_board(edges)), h_index(1, 3))

    _test_encode_decode()
    _test_get_available_actions_encoded()
