from dotsandboxes import DotsAndBoxesEngine
from config import *
from util import get_boxes_adjacent_to_edge, bit_count_box_edges, bit_makes_third_edge 
from search.TranspositionTable import TranspositionTable


# Search Extension 

def complete_extension(out, action: Action):
    if out['is_box_completed']:
        return True

def give_away_extension(out, action: Action):
    
    board = out['state']['board']
    b_list = get_boxes_adjacent_to_edge(action)
    for box in b_list:
        if (bit_count_box_edges(board, box) == 3):
            return True
    return False


# Move Ordering

def default_move_ordering(actions: list[Action], eng: DotsAndBoxesEngine, tt: TranspositionTable, depth:int, root_player:int):
    return actions

def _complete_box(board: Board, action: Action) -> bool:
    for box in get_boxes_adjacent_to_edge(action):
        if bit_count_box_edges(board, box) == 3:
            # 지금 두면 박스가 완성됨
            return True
        
def _make_double_cross(edges, action):
    c, r, d = action
    h, v = edges
    b_list = get_boxes_adjacent_to_edge(c, r, d)
    if len(b_list) == 2:
        return (bit_count_box_edges(h, v, b_list[0][0], b_list[0][1]) == 2) and (bit_count_box_edges(h, v, b_list[1][0], b_list[1][1]) == 2)
    else:
        return False
## Move_Ordering
def move_ordering(actions: list[Action], eng: DotsAndBoxesEngine, tt: TranspositionTable, depth:int, root_player:int):
    
    state = eng.get_state()
    edges = state.board

    ranked = []

    forced = []
    safe = []
    for a in actions:
        if _complete_box(edges, a):
            forced.append(a)
        
        if not bit_makes_third_edge(edges, a):
            safe.append(a)

    rest = actions
    rest = [a for a in rest if a not in forced]
    rest = [a for a in rest if a not in safe]

    ranked = forced + safe + rest

    return ranked

def move_ordering_v2(actions: list[Action], eng: DotsAndBoxesEngine, depth:int, root_player:int):
    
    state = eng.get_state()
    edges = state["edges"]

    ranked = []

    forced = []
    double_cross = []
    safe = []
    for a in actions:
        if _complete_box(edges, a):
            forced.append(a)
        
        if _make_double_cross(edges, a):
            double_cross.append(a)

        if not bit_makes_third_edge(edges, a):
            safe.append(a)

    rest = actions
    rest = [a for a in rest if a not in forced]
    rest = [a for a in rest if a not in double_cross]
    rest = [a for a in rest if a not in safe]

    ranked = forced + safe + double_cross + rest

    return ranked
