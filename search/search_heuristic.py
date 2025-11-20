from dotsandboxes import DotsAndBoxesEngine
from .TranspositionTable import TranspositionTable
from util import get_boxes_adjacent_to_edge, count_box_edge, makes_third_edge 

def complete_extension(info, out, action):
    if out['is_box_completed']:
        return True

def give_away_extension(info, out, action):
    
    h, v = out['state']['edges']
    c, r, d= action
    b_list = get_boxes_adjacent_to_edge(c, r, d)
    for br, bc in b_list:
        if (count_box_edge(h, v, br, bc) == 3):
            return True
    return False

def compelete_box(edges, action) -> bool:
    c, r, d = action
    hb, vb = edges
    for (bc, br) in get_boxes_adjacent_to_edge(c, r, d):
        if count_box_edge(hb, vb, bc, br) == 3:
            # 지금 두면 박스가 완성됨
            return True
        
def make_double_cross(edges, action):
    c, r, d = action
    h, v = edges
    b_list = get_boxes_adjacent_to_edge(c, r, d)
    if len(b_list) == 2:
        return (count_box_edge(h, v, b_list[0][0], b_list[0][1]) == 2) and (count_box_edge(h, v, b_list[1][0], b_list[1][1]) == 2)
    else:
        return False
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
        
        if not makes_third_edge(edges, a):
            safe.append(a)

            

    rest = actions
    rest = [a for a in rest if a not in forced]
    rest = [a for a in rest if a not in safe]

    ranked = forced + safe + rest

    return ranked

def move_ordering_v2(actions, eng: DotsAndBoxesEngine, tt: TranspositionTable, depth:int, root_player:int):
    
    state = eng.get_state()
    edges = state["edges"]
    cur_player = state["cur_player"]
    score = state["score"]

    ranked = []

    forced = []
    double_cross = []
    safe = []
    for a in actions:
        if compelete_box(edges, a):
            forced.append(a)
        
        if make_double_cross(edges, a):
            double_cross.append(a)

        if not makes_third_edge(edges, a):
            safe.append(a)

    rest = actions
    rest = [a for a in rest if a not in forced]
    rest = [a for a in rest if a not in double_cross]
    rest = [a for a in rest if a not in safe]

    ranked = forced + safe + double_cross + rest

    return ranked
