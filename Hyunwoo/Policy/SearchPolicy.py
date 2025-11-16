from DotsAndBoxes import DnBEnv
from .BasePolicy import BasePolicy
from DotsAndBoxes import DotsAndBoxesEngine
from typing import Any, Callable, Iterable, Tuple, Optional, NamedTuple, List
from Util.DnB_Engine_Util import *
from Search import BaseSearchEngine, AlphaBetaSearch, TranspositionTable, TTEntry
from .Scheduler import *
import numpy as np

Action = List[int]





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

    def get_log(self):
        log = self.SearchEngine.get_log()
        self.SearchEngine.reset_log()
        return log
def main():
    AlphaBetaSearch = AlphaBetaSearch(evaluate=evaluate, move_ordering=None, depth=3)
    SearchPolicy(SearchEngine=AlphaBetaSearch)
