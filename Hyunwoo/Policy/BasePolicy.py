from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

import importlib

from DotsAndBoxes import DnBEnv
import time

class TimeManager():
    def __init__(self):
        self.total_budget = 24.0
        self.used_time = 0.0
        self._move_start = None

    def remaining(self):
        return max(0.0, self.total_budget - self.used_time)
    
    def start_move(self):
        self._move_start = time.perf_counter()
    
    def end_move(self):
        dt = time.perf_counter() - self._move_start
        self.used_time += dt
        return dt
    
    def reset(self):
        self.total_budget = 24.0
        self.used_time = 0.0

class BasePolicy():
    def __init__(self):
        ## 필요한거 있으면 추가
        self.time_manager = TimeManager()
        pass

    def get_action(self, observation, info, env, time_manager:TimeManager):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음
        raise NotImplementedError

    def get_log(self):
        return None

## 예시 정책
class RandomPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def get_action(self, observation, info, env, time_manager:TimeManager):
        action_mask = info['action_mask']
        action = env.action_space.sample()
        # info['action_mask']가 True인 액션은 이미 선택된 액션이므로 다시 샘플링
        while action_mask[action[0], action[1], action[2]]:
            action = env.action_space.sample()

        return action, None


class FixedOrderPolicy(BasePolicy):
    def __init__(self, n_box):
        super().__init__()
        self.n_box = n_box
        self.action_order = []
        for ori in range(2):
            for r in range(n_box + 1):
                for c in range(n_box + 1):
                    if (ori == 0 and c == n_box) or (ori == 1 and r == n_box):
                        continue
                    self.action_order.append((c, r, ori))
        self.current_index = 0


    def get_action(self, observation, info, env, time_manager:TimeManager):
        action = self.action_order[self.current_index]
        while info['action_mask'][action[0], action[1], action[2]]:
            self.current_index = (self.current_index + 1) % len(self.action_order)
            action = self.action_order[self.current_index]

        self.current_index = (self.current_index + 1) % len(self.action_order)
        
        return action, None
