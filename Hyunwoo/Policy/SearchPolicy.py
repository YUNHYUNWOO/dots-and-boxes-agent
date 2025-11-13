from DotsAndBoxes import DnBEnv
from .BasePolicy import BasePolicy
from DotsAndBoxes import DotsAndBoxesEngine
from typing import Any, Callable, Iterable, Tuple, Optional, NamedTuple, List
from Util.DnB_Engine_Util import *
from Search import BaseSearchEngine, AlphaBetaSearch, TranspositionTable, TTEntry
from .Scheduler import *
import numpy as np

Action = List[int]

def _adjacent_boxes(r: int, c: int, d: int) -> List[Tuple[int, int]]:
    boxes = []
    if d == H:
        if 0 <= r - 1 < N_BOX: boxes.append((r - 1, c))
        if 0 <= r < N_BOX:     boxes.append((r, c))
    else:
        if 0 <= c - 1 < N_BOX: boxes.append((r, c - 1))
        if 0 <= r < N_BOX and 0 <= c < N_BOX: boxes.append((r, c))
    return boxes

def _box_edge_count(hb: int, vb: int, br: int, bc: int) -> int:
    cnt = 0
    # H(br,bc), H(br+1,bc)
    if (hb >> h_index(br, bc)) & 1:       cnt += 1
    if (hb >> h_index(br + 1, bc)) & 1:   cnt += 1
    # V(br,bc), V(br,bc+1)
    if (vb >> v_index(br, bc)) & 1:       cnt += 1
    if (vb >> v_index(br, bc + 1)) & 1:   cnt += 1
    return cnt


def _makes_third_edge(eng: DotsAndBoxesEngine, a) -> bool:
    """액션이 인접 박스 중 '3번째 엣지'를 만들어서 상대에게 4번째를 헌납할 위험인지 체크."""
    c, r, d = a
    hb, vb = eng.h_bits, eng.v_bits
    for (br, bc) in _adjacent_boxes(r, c, d):
        if _box_edge_count(hb, vb, br, bc) == 2:
            # 지금 두면 3이 됨 (위험수)
            return True
    return False

def evaluate(eng: DotsAndBoxesEngine, root_player: int) -> int:
    """아주 단순한 평가식: 점수차 크게 + '위험수(3-edge 만드는 수)'는 페널티."""
    me, opp = root_player, 1 - root_player
    score_term = (eng.score[me] - eng.score[opp]) * 100

    moves = get_legal_actions(eng.get_state()['edges'])
    bad_moves = sum(1 for m in moves if _makes_third_edge(eng, m))
    # bad_moves가 적을수록 좋다

    return score_term - bad_moves

def evaluate_rel(eng: DotsAndBoxesEngine) -> int:
    moves = get_legal_actions(eng.get_state()['edges'])
    bad_moves = sum(1 for m in moves if _makes_third_edge(eng, m))
    # bad_moves가 적을수록 좋다
    bad_moves /= 100
    return -bad_moves

# ===============================================
# Todo Chain analysis
# eng.get_state()
    # {
    #     "edges": [self.h_bits, self.v_bits],
    #     "cur_player": self.cur_player,
    #     "score": self.score[:],
    # }

def evaluate_chain(eng: DotsAndBoxesEngine) -> int:

    moves = get_legal_actions(eng.get_state()['edges'])
    bad_moves = sum(1 for m in moves if _makes_third_edge(eng, m))
    # bad_moves가 적을수록 좋다
    bad_moves /= 100
    return -bad_moves


## Move_Ordering
def move_ordering(actions, eng: DotsAndBoxesEngine, tt: TranspositionTable, depth:int, root_player:int):
    scored = []
    for a in actions:
        out = eng.apply_action(a)
        maximizing = (eng.get_state()['cur_player'] == root_player)

        player_before = eng.cur_player
        key = tt.key_from(eng, maximizing)
        ent = tt._t.get(key)
        immediate_val = len(out["completed_boxes"])
        sign = 1 if (root_player == player_before) else -1
        
        eng.undo_action(a, out["completed_boxes"], player_before)

        if ent is not None:
            scored.append((sign * immediate_val + ent.value, a))
        else:
            scored.append((-100, a))  # 기본값

    # Max node면 높은 값부터, Min node면 낮은 값부터
    scored.sort(reverse=maximizing)
    return [a for _, a in scored]

# =======================================

class SearchPolicy(BasePolicy):
    def __init__(self, SearchEngine:BaseSearchEngine, config_schedule: Dict):
        ## 필요한거 있으면 추가
        self.eng = DotsAndBoxesEngine()
        self.SearchEngine = SearchEngine
        self.config_schedule = config_schedule

    def get_config(self, t):
        config = {}
        for k, v in self.config_schedule.items():
            if isinstance(v, BaseScheduler):
                # BaseScheduler의 하위 클래스면 .value(t)
                config[k] = v.value(t)
            else:
                # 상수면 그대로 사용
                config[k] = v

        return config
    
    def get_action(self, observation, info, env):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음
        b_edges = encode_Edges(observation['edges'])

        state = {
            'edges': b_edges,
            'cur_player': observation['cur_player'],
            'score': observation['score']
        }
        t = 60 - np.sum(info['action_mask'] == False)
        config = self.get_config(t)

        self.SearchEngine.configure(**config)
        self.eng.set_state(state)

        best_action, best_val = self.SearchEngine.search(eng=self.eng, state=state)
        
        return best_action, best_val

def main():
    AlphaBetaSearch = AlphaBetaSearch(evaluate=evaluate, move_ordering=None, depth=3)
    SearchPolicy(SearchEngine=AlphaBetaSearch)
