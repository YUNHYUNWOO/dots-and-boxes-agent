import numpy as np

from config import *
from dotsandboxes import DotsAndBoxesEngine
from util.bit_dnb_util import (encode_board, decode_bitboard)
from util.time_manager import TimeManager
from search import BaseSearchEngine

from util.scheduler import *
from .basepolicy import BasePolicy, TimeManager

# =======================================

class SearchPolicy(BasePolicy):
    def __init__(self, search_engine:BaseSearchEngine, config_schedule: dict):
        ## 필요한거 있으면 추가
        self.eng = DotsAndBoxesEngine()
        self.search_engine = search_engine
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
        bitBoard = encode_board(observation['board'])

        state = DnBEngineState(board=bitBoard,
                               cur_player=observation['cur_player'])
        self.eng.set_state(state)

        def get_t(board:Board):
            t = 0
            for c in range(N):
                for r in range(N):
                    for d in range(2):
                        if board[c][r][d] == 1:
                            t += 1
            return t
        t = get_t(observation['board'])
        config = self.get_config(t)

        self.search_engine.configure(**config)

        best_action, _ = self.search_engine.search(eng=self.eng, state=state, time_manager=time_manager)
        
        return best_action

    def get_log(self):
        log = self.search_engine.get_log()
        self.search_engine.reset_log()
        return log

