from Util import get_connected_Components, init_box_data, get_cv, get_legal_actions, classify_component
from DotsAndBoxes import DotsAndBoxesEngine
from Util.DnB_Engine_Util import *
from Search import BaseSearchEngine, AlphaBetaSearch, TranspositionTable, TTEntry


def _adjacent_boxes(c: int, r: int, d: int) -> List[Tuple[int, int]]:
    boxes = []
    if d == H:
        if 0 <= r - 1 < N_BOX: boxes.append((c, r - 1))
        if 0 <= r < N_BOX:     boxes.append((c, r))
    else:
        if 0 <= c - 1 < N_BOX: boxes.append((c - 1, r))
        if 0 <= r < N_BOX and 0 <= c < N_BOX: boxes.append((c, r))
    return boxes

def _box_edge_count(hb: int, vb: int, bc: int, br: int) -> int:
    cnt = 0
    # H(br,bc), H(br+1,bc)
    if (hb >> h_index(bc, br)) & 1:       cnt += 1
    if (hb >> h_index(bc, br + 1)) & 1:   cnt += 1
    # V(br,bc), V(br,bc+1)
    if (vb >> v_index(bc, br)) & 1:       cnt += 1
    if (vb >> v_index(bc + 1, br)) & 1:   cnt += 1
    return cnt

def _makes_third_edge(edges, action) -> bool:
    """액션이 인접 박스 중 '3번째 엣지'를 만들어서 상대에게 4번째를 헌납할 위험인지 체크."""
    c, r, d = action
    h, v = edges
    for (bc, br) in _adjacent_boxes(c, r, d):
        if _box_edge_count(h, v, bc, br) == 2:
            # 지금 두면 3이 됨 (위험수)
            return True
    return False

def evaluate_rel(eng: DotsAndBoxesEngine) -> int:
    moves = get_legal_actions(eng.get_state()['edges'])
    edges = eng.get_state()['edges']

    bad_moves = sum(1 for m in moves if _makes_third_edge(edges, m))
    # bad_moves가 적을수록 좋다
    bad_moves /= 100
    return -bad_moves


def evaluate_cv(eng):
    edges = eng.get_state()['edges']

    adj, external_open, is_candidate = init_box_data(edges)
    comps = get_connected_Components(adj, is_candidate)
    comps = classify_component(comps, adj)
    return get_cv(comps)

def compelete_box(edges, action) -> bool:
    c, r, d = action
    hb, vb = edges
    for (bc, br) in _adjacent_boxes(c, r, d):
        if _box_edge_count(hb, vb, bc, br) == 3:
            # 지금 두면 박스가 완성됨
            return True

# def opens_chain(edges, action) -> bool:
#     adj, external_open, is_candidate = init_box_data(edges)
#     comps = get_connected_Components(adj, is_candidate)

#     for comp in comps:       


## Move_Ordering
def move_ordering(actions, eng: DotsAndBoxesEngine, tt: TranspositionTable, depth:int, root_player:int):
    
    state = eng.get_state()
    edges = state["edges"]
    cur_player = state["cur_player"]
    score = state["score"]

    ranked = []

    forced = []
    safe = []
    for a in actions:
        if compelete_box(edges, a):
            forced.append(a)
        
        if not _makes_third_edge(edges, a):
            safe.append(a)

    rest = actions
    rest = [a for a in rest if a not in forced]
    rest = [a for a in rest if a not in safe]

    ranked = forced + safe + rest

    return ranked



        