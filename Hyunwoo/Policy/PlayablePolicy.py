from DotsAndBoxes import DotsAndBoxes, DnBEnv
from .BasePolicy import BasePolicy, TimeManager
import pygame

class PlayablePolicy(BasePolicy):


    def get_policy(self, t):
        # print(self.policy_scheduler.get_config())
        return self.policy_scheduler.value(t)
    
    def get_action(self, observation, info, env: DnBEnv, time_manager: TimeManager):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음

        DnB = env.DnB
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    e = DnB.find_hover_edge(event.pos)
                    if e is not None:
                        waiting = False
                        break
        d, r, c = e
        action = [c, r, 0 if d == 'H' else 1]
        return action, None

        