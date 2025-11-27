from config import *
from dotsandboxes import DotsAndBoxesEngine
from util import (
    bit_get_legal_actions,
    bit_makes_third_edge,
    bit_is_box_complete,
    bit_count_box_edges,

    init_box_data,
    init_box_data_for_components,
    get_connected_components,
    classify_component,
    get_long_chain,
    compute_chain_risk
)

Score = float

def evaluate_rel(eng: DotsAndBoxesEngine) -> Score:
    actions = bit_get_legal_actions(eng.get_state().board)
    board = eng.get_state().board

    bad_moves = sum(1 for a in actions if bit_makes_third_edge(board, a))
    # bad_moves가 적을수록 좋다
    bad_moves /= 100
    return -bad_moves

def evaluate_relv2(eng: DotsAndBoxesEngine) -> Score:
    board = eng.get_state().board
    actions = bit_get_legal_actions(board)

    bad_moves = sum(1 for a in actions if bit_makes_third_edge(board, a))
    # bad_moves가 적을수록 좋다

    cnt = 0
    for br in range(N_BOX):
        for bc in range(N_BOX):
            if bit_is_box_complete(board, (bc, br)): 
                cnt += 1

    if (cnt % 2) == 0:
        bad_moves += 5

    bad_moves /= 100
    return -bad_moves

def evaluate_chain_added(eng: DotsAndBoxesEngine) -> int:
    board = eng.get_state().board
    actions = bit_get_legal_actions(board)


    bad_moves = 0
    for action in actions:
        c, r, d = action
        if d == H:
            if 0 <= r < N_BOX:
                if bit_count_box_edges(board, (c, r)) == 2:
                    bad_moves += 1
                    continue
            if 0 <= r - 1 < N_BOX:
                if bit_count_box_edges(board, (c, r-1)) == 2:
                    bad_moves += 1
        else:
            if 0 <= r < N_BOX:
                if 0 <= c < N_BOX and bit_count_box_edges(board, (c, r)) == 2:
                    bad_moves += 1
                    continue
                if 0 < c and bit_count_box_edges(board, (c-1, r)) == 2:
                    bad_moves += 1

    used_edge = 0
    h, v = board
    for i in range(30):
        if ((h >> i) & 1): used_edge += 1
        if ((v >> i) & 1): used_edge += 1

    if (used_edge > 25):
        cnt = 0
        for br in range(N_BOX):
            for bc in range(N_BOX):
                if (h >> (br * (N - 1) + bc) & 1) & (h >> ((br + 1) * (N - 1) + bc) & 1) & (v >> (br * N + bc) & 1) & (v >> (br * N + bc + 1) & 1): 
                    cnt += 1
        
        if (cnt % 2) == 0:
            adj, is_candidate = init_box_data(board)
            comps = get_connected_components(adj, is_candidate)
            comps = classify_component(comps, adj)

            bad_moves += 4 + get_long_chain(comps) * 0.6

    bad_moves /= 100

    return -bad_moves

def evaluate_chain_aware(eng: DotsAndBoxesEngine) -> Score:
    """
    현재 턴(cur_player) 입장에서:
    - 점수는 최대화
    - safe move는 많을수록 좋음
    - 체인/루프 길이가 길어질수록 나쁨
    """

    board = eng.get_state().board

    # 2) safe move 개수
    legal = bit_get_legal_actions(board)
    safe_moves = [a for a in legal if not bit_makes_third_edge(board, a)]
    v_safe = len(safe_moves)

    # 3) 체인/루프 길이 기반 risk
    chain_risk, largest = compute_chain_risk(board)

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

def evaluate_comps(eng: DotsAndBoxesEngine) -> Score:
    board = eng.get_state().board
    adj, is_candidate = init_box_data_for_components(board)
    comps = get_connected_components(adj, is_candidate)
    return len(comps)
# def opens_chain(edges, action) -> bool:
#     adj, external_open, is_candidate = init_box_data(edges)
#     comps = get_connected_Components(adj, is_candidate)

#     for comp in comps:       