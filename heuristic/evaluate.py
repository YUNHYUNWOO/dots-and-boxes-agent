"""Heuristic evaluation functions for Dots and Boxes states."""

from config import H, N, N_BOX, Action
from dotsandboxes import DotsAndBoxesEngine
from util.bit_dnb_util import (
    bit_count_box_edges,
    bit_get_legal_actions,
    bit_makes_third_edge,
    bit_count_edges
)
from util.chain import (
    classify_component,
    get_connected_components,
    get_chain_risk,
    init_box_data,
)

Score = float

def evaluate_default(eng: DotsAndBoxesEngine) -> Score:
    """Penalize moves that create third edges on adjacent boxes."""
    return 0

def evaluate_bad_moves(eng: DotsAndBoxesEngine) -> Score:
    """Penalize moves that create third edges on adjacent boxes."""

    board = eng.get_state().board
    actions = bit_get_legal_actions(board)
   
    bad_moves = sum(1 for a in actions if bit_makes_third_edge(board, a)) 
    return -(bad_moves / 100)
    # return 0


def evaluate_chain(eng: DotsAndBoxesEngine) -> int:
    """Penalize moves that extend chains into risky third edges."""

    board = eng.get_state().board
    actions = bit_get_legal_actions(board)

    bad_moves = sum(1 for a in actions if bit_makes_third_edge(board, a)) 

    t = bit_count_edges(board)

    chain_risk = 0
    if t > 25:
        cnt = 0
        for br in range(N_BOX):
            for bc in range(N_BOX):
                if bit_count_box_edges(board, (bc, br)) == 3:
                    cnt += 1

        # when you have control. doesn't consider chain_risk 
        if (cnt % 2) == 0:
            adj, is_candidate = init_box_data(board)
            comps = get_connected_components(adj, is_candidate)
            comps = classify_component(comps, adj)

            chain_risk += 4 + get_chain_risk(comps) * 0.6

    return -((bad_moves + chain_risk)/ 100)