import time
import random 

from config import *
from dotsandboxes import DnBEnv
from util.time_manager import TimeManager

class BasePolicy():
    def __init__(self):
        ## 필요한거 있으면 추가
        self.time_manager = TimeManager()
        pass

    def get_action(self, observation, time_manager:TimeManager)-> Action:
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

    def get_action(self, observation, time_manager:TimeManager):
        
        board = observation['board']
        def sample_action() -> Action:
            c, r, d = random.randrange([0, 6]), random.randrange([0, 6]), random.randrange([0, 2])
            if (d == H and c == N_BOX) or (d == V and r == N_BOX):
                return sample_action()
            return (c, r, d)

        action = sample_action()
        while board[action[0], action[1], action[2]]:
            action = sample_action()

        return action


class FixedOrderPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
        self.action_order = []
        for ori in range(2):
            for r in range(N_BOX + 1):
                for c in range(N_BOX + 1):
                    if (ori == 0 and c == N_BOX) or (ori == 1 and r == N_BOX):
                        continue
                    self.action_order.append((c, r, ori))
        self.cur_index = 0


    def get_action(self, observation, time_manager:TimeManager):
        action = self.action_order[self.cur_index]

        while observation['board'][action[0]][action[1]][action[2]]:
            self.cur_index = (self.cur_index + 1) % len(self.action_order)
            action = self.action_order[self.cur_index]

        self.cur_index = (self.cur_index + 1) % len(self.action_order)
        
        return action
