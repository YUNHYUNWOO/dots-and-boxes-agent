from .BasePolicy import BasePolicy, TimeManager
from .Scheduler import PiecewiseConstantScheduler
import numpy as np


class MixedPolicy(BasePolicy):
    def __init__(self, policy_scheduler: PiecewiseConstantScheduler):
        self.policy_scheduler = policy_scheduler

    def get_policy(self, t):
        # print(self.policy_scheduler.get_config())
        return self.policy_scheduler.value(t)
    
    def get_action(self, observation, info, env, time_manager:TimeManager):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음

        t = 60 - np.sum(info['action_mask'] == False)
        policy = self.get_policy(t)
        
        return policy.get_action(observation, info, env, time_manager)