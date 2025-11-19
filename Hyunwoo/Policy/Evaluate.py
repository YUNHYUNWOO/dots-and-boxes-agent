from Util import get_connected_Components, init_box_data, get_cv, get_legal_actions, classify_component
from DotsAndBoxes import DotsAndBoxesEngine
from Util.DnB_Engine_Util import *
from Search import BaseSearchEngine, AlphaBetaSearch, TranspositionTable, TTEntry
from Util import makes_third_edge



def evaluate_rel(eng: DotsAndBoxesEngine) -> int:
    moves = get_legal_actions(eng.get_state()['edges'])
    edges = eng.get_state()['edges']

    bad_moves = sum(1 for m in moves if makes_third_edge(edges, m))
    # bad_moves가 적을수록 좋다
    bad_moves /= 100
    return -bad_moves


def evaluate_cv(eng):
    edges = eng.get_state()['edges']

    adj, external_open, is_candidate = init_box_data(edges)
    comps = get_connected_Components(adj, is_candidate)
    comps = classify_component(comps, adj)
    return get_cv(comps)

def evaluate_relv2(eng: DotsAndBoxesEngine) -> int:
    moves = get_legal_actions(eng.get_state()['edges'])
    edges = eng.get_state()['edges']

    bad_moves = sum(1 for m in moves if makes_third_edge(edges, m))
    # bad_moves가 적을수록 좋다

    cnt = 0
    for r in range(N_BOX):
        for c in range(N_BOX):
            if is_box_complete(edges[0], edges[1], c, r): 
                cnt += 1

    if (cnt % 2) == 0:
        bad_moves += 5

    bad_moves /= 100
    return -bad_moves


        
# def opens_chain(edges, action) -> bool:
#     adj, external_open, is_candidate = init_box_data(edges)
#     comps = get_connected_Components(adj, is_candidate)

#     for comp in comps:       