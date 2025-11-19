from Util import get_connected_Components, init_box_data, get_cv, get_legal_actions, classify_component, makes_third_edge, adjacent_boxes, boxes_adjacent_to_edge
from DotsAndBoxes import DotsAndBoxesEngine
from Util.DnB_Engine_Util import *
from Search import BaseSearchEngine, AlphaBetaSearch, TranspositionTable, TTEntry

def evaluate_rel(eng: DotsAndBoxesEngine) -> int:
    moves = get_legal_actions(eng.get_state()['edges'])
    edges = eng.get_state()['edges']

    bad_moves = sum(1 for m in moves if makes_third_edge(edges, m))
    # bad_moves가 적을수록 좋다
    bad_moves /= 100
    return -bad_moves

def evaluate_relv2(eng: DotsAndBoxesEngine) -> int:
    moves = get_legal_actions(eng.get_state()['edges'])
    edges = eng.get_state()['edges']

    bad_moves = sum(1 for m in moves if makes_third_edge(edges, m))
    # bad_moves가 적을수록 좋다

    cnt = 0
    for br in range(N_BOX):
        for bc in range(N_BOX):
            if (edges[0] >> (br * (N - 1) + bc) & 1) & (edges[0] >> ((br + 1) * (N - 1) + bc) & 1) & (edges[1] >> (br * N + bc) & 1) & (edges[1] >> (br * N + bc + 1) & 1): 
                cnt += 1
    
    if (cnt % 2) == 0:
        bad_moves += 6

    bad_moves /= 100

    return -bad_moves

def evaluate_relv3(eng: DotsAndBoxesEngine) -> int:
    edges = eng.get_state()['edges']

    cnt = 0
    for br in range(N_BOX):
        for bc in range(N_BOX):
            if (edges[0] >> (br * (N - 1) + bc) & 1) & (edges[0] >> ((br + 1) * (N - 1) + bc) & 1) & (edges[1] >> (br * N + bc) & 1) & (edges[1] >> (br * N + bc + 1) & 1): 
                cnt += 1
    
    if (cnt % 2) == 0:
        return -cnt * 5

    return 5


def evaluate_cv(eng):
    edges = eng.get_state()['edges']

    adj, external_open, is_candidate, cnt_safe_box = init_box_data(edges)
    comps = get_connected_Components(adj, is_candidate)
    comps = classify_component(comps, adj)
    return get_cv(comps)

# def opens_chain(edges, action) -> bool:
#     adj, external_open, is_candidate = init_box_data(edges)
#     comps = get_connected_Components(adj, is_candidate)

#     for comp in comps:       



        