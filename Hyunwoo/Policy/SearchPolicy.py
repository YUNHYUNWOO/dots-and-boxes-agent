from DotsAndBoxes import DnBEnv
from .BasePolicy import BasePolicy
from DotsAndBoxes import DotsAndBoxesEngine
from typing import Any, Callable, Iterable, Tuple, Optional, NamedTuple, List
from Util.DnB_Engine_Util import *
from Search import BaseSearchEngine, AlphaBetaSearch

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


class Search_Policy(BasePolicy):
    def __init__(self, SearchEngine:BaseSearchEngine):
        ## 필요한거 있으면 추가
        self.eng = DotsAndBoxesEngine()
        self.SearchEngine = SearchEngine


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
        self.eng.set_state(state)
        
        best_action, best_val = self.SearchEngine.search(eng=self.eng, state=state)
        
        return best_action
    

def main():
    AlphaBetaSearch = AlphaBetaSearch(evaluate=evaluate, move_ordering=None, depth=3)
    Search_Policy(SearchEngine=AlphaBetaSearch)
