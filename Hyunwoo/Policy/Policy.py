from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

import importlib

from DotsAndBoxes import DotsAndBoxes, DnBEnv

class Policy():
    def __init__(self):
        ## 필요한거 있으면 추가
        pass


    def get_action(self, observation, info, env):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음
        raise NotImplementedError


## 예시 정책
class RandomPolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_action(self, observation, info, env):
        action_mask = info['action_mask']
        action = env.action_space.sample()
        # info['action_mask']가 True인 액션은 이미 선택된 액션이므로 다시 샘플링
        while action_mask[action[0], action[1], action[2]]:
            action = env.action_space.sample()

        return action


class FixedOrderPolicy(Policy):
    def __init__(self, n_box):
        super().__init__()
        self.n_box = n_box
        self.action_order = []
        for ori in range(2):
            for r in range(n_box + 1):
                for c in range(n_box + 1):
                    if (ori == 0 and c == n_box) or (ori == 1 and r == n_box):
                        continue
                    self.action_order.append((ori, r, c))
        self.current_index = 0


    def get_action(self, observation, info, env):
        action = self.action_order[self.current_index]
        while info['action_mask'][action[0], action[1], action[2]]:
            self.current_index = (self.current_index + 1) % len(self.action_order)
            action = self.action_order[self.current_index]

        self.current_index = (self.current_index + 1) % len(self.action_order)
        
        return action



if __name__ == "__main__":

    n_box = 5
    env = DnBEnv(render_mode='human', n_box=n_box)
    
    policy = FixedOrderPolicy(n_box=n_box)
    # policy = RandomPolicy()
    SimulateEpisode(env, policy, verbose=True)
