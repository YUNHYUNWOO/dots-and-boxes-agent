
import numpy as np

from config import *

from .basepolicy import BasePolicy
from util.scheduler import PiecewiseConstantScheduler
from util.time_manager import TimeManager


class MixedPolicy(BasePolicy):
    def __init__(self, policy_scheduler: PiecewiseConstantScheduler):
        self.policy_scheduler = policy_scheduler

    def get_policy(self, t):
        # print(self.policy_scheduler.get_config())
        return self.policy_scheduler.value(t)
    
    def get_action(self, observation:dict, time_manager:TimeManager):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음

        def get_t(board:Board):
            t = 0
            for c in range(N):
                for r in range(N):
                    for d in range(N):
                        t += 1
            return t
        t = get_t(observation['board'])
        
        policy = self.get_policy(t)
        
        return policy.get_action(observation, time_manager)