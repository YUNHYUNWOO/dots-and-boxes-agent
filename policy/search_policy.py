import numpy as np

from dotsandboxes import DnBEnv, DotsAndBoxesEngine
from util import *
from search import BaseSearchEngine

from .scheduler import *
from .basepolicy import BasePolicy, TimeManager

# =======================================

class SearchPolicy(BasePolicy):
    def __init__(self, SearchEngine:BaseSearchEngine, config_schedule: dict):
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
    
    def get_action(self, observation:dict, time_manager:TimeManager):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음
        b_edges = encode_board(observation['edges'])

        state = {
            'edges': b_edges,
            'cur_player': observation['cur_player'],
            'score': observation['score']
        }

        def get_t(board:Board):
            t = 0
            for c in range(N):
                for r in range(N):
                    for d in range(N):
                        t += 1
            return t
        t = get_t(observation['board'])
        config = self.get_config(t)

        self.SearchEngine.configure(**config)
        self.eng.set_state(state)

        best_action, _ = self.SearchEngine.search(eng=self.eng, state=state, time_manager=time_manager)
        
        return best_action

    def get_log(self):
        log = self.SearchEngine.get_log()
        self.SearchEngine.reset_log()
        return log

