from dotsandboxes import DotsAndBoxesEngine
from util import *
from search import BaseSearchEngine, AlphaBetaSearch


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

def evaluate_relv2(eng: DotsAndBoxesEngine):
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

def evaluate_relv3(eng: DotsAndBoxesEngine) -> int:
    moves = get_legal_actions(eng.get_state()['edges'])
    h, v = eng.get_state()['edges']

    bad_moves = 0
    for c, r, d in moves:
        tmp = 0
        if d == H:
            if 0 <= r < N_BOX:
                if count_box_edge(h, v, c, r) == 2:
                    tmp = 1
            if 0 <= r - 1 < N_BOX:
                if count_box_edge(h, v, c, r - 1) == 2:
                    tmp = 1
        else:
            if 0 <= r < N_BOX:
                if 0 <= c < N_BOX and count_box_edge(h, v, c, r) == 2:
                    tmp = 1
                elif c - 1 < N_BOX and count_box_edge(h, v, c, r) == 2:
                    tmp = 1
        bad_moves += tmp

    adj, external_open, is_candidate = init_box_data([h, v])
    comps = get_connected_Components(adj, is_candidate)
    comps = classify_component(comps, adj)
    bad_moves += get_long_chain(comps) * 0.5

    bad_moves /= 100

    return -bad_moves

def evaluate_chain_aware(eng) -> float:
    """
    현재 턴(cur_player) 입장에서:
    - 점수는 최대화
    - safe move는 많을수록 좋음
    - 체인/루프 길이가 길어질수록 나쁨
    """
    state = eng.get_state()
    edges = state["edges"]

    # 2) safe move 개수
    legal = get_legal_actions(edges)
    safe_moves = [a for a in legal if not makes_third_edge(edges, a)]
    v_safe = len(safe_moves)

    # 3) 체인/루프 길이 기반 risk
    chain_risk, largest = compute_chain_risk(edges)

    # 4) 가중치 (필요하면 나중에 튜닝)
    W_SAFE  =  0.01   # safe move 조금 선호
    W_CHAIN =  0.05   # 체인 길이 위험도
    W_REGION = 0.05

    value = (
        W_SAFE  * v_safe  -
        W_CHAIN * chain_risk -
        W_REGION * largest
    )
    return value

def evaluate_comps(eng: DotsAndBoxesEngine):
    edges = eng.get_state()['edges']
    adj, external_open, is_candidate = init_box_data_for_components(edges)
    comps = get_connected_Components(adj, is_candidate)
    return len(comps)
# def opens_chain(edges, action) -> bool:
#     adj, external_open, is_candidate = init_box_data(edges)
#     comps = get_connected_Components(adj, is_candidate)

#     for comp in comps:       