from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from DnB import DotsAndBoxes, get_render_desc, draw_board



# DnB 환경의 상태, 행동을 DnB Env의 형태로 변환
def interpret_edges(edges):
    def map_func(e):
        if e == 0:
            return -1
        elif e == None:
            return 0
        else:
            return e

    i_edges = [[map_func(e) for e in r] for r in edges]

    return i_edges


def interpret_box_owner(box_owner):
    def map_func(box):
        if box == 0:
            return -1
        elif box == None:
            return 0
        else :
            return box 
        
    for r in range(len(box_owner)):
        for c, e in enumerate(box_owner[r]):
            if e == 0:
                box_owner[r][c] = -1
    i_box_owner = [[map_func(box) for box in r] for r in box_owner]
    return i_box_owner


def interpret_action(action):
    ori = 'H' if action[0] == 0 else 'V'
    r, c = map(int, action[1:])
    return (ori, r, c)

class DnBEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, n_box = 5):

        
        self.n_box = n_box # grid size
        self.DnB = None # Game 인스턴스, 초기화는 reset에서 수행
        self.window_size = 512 #The size of the Pygame window

        # Observation space 정의
        # h_edges: 가로 선 6x5  (0: 없음, 1: 있음)
        # v_edges: 세로 선 5x6  (0: 없음, 1: 있음)
        # box_owner: 상자 소유자 5x5 (-1: 유저1, 0: 없음, 1: 유저2)]

        self.observation_space = spaces.Dict(
            {
                "h_edges": spaces.Box(-1, 1, shape=(6,5), dtype=int),
                "v_edges": spaces.Box(-1, 1, shape=(5,6), dtype=int),
                "box_owner": spaces.Box(-1, 1, shape=(5,5), dtype=int)
            }
        )
        
        self.h_board = np.zeros(shape=(6,5), dtype=bool)
        self.v_board = np.zeros(shape=(5,6), dtype=bool)


        # action space 정의
        # action: (edge_type, row, col)
        # edge_type: 0 (h_edge), 1 (v_edge)
        # row: 0~5 (h_edge), 0~4 (v_edge) 
        # col: 0~4 (h_edge), 0~5 (v_edge)
        # 불가능한 행동은 action mask로 후에 처리
        self.action_space = spaces.MultiDiscrete(nvec=[2,6,6], dtype=int)
        
        self.action_mask = None


        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        
        self.screen = None
        self.clock = None
        self.fonts = None


    def _get_obs(self):
        # potential Copy issue
        obs = {
            'h_edges': interpret_edges(self.DnB.h_edges),
            'v_edges': interpret_edges(self.DnB.v_edges),
            'box_owner': interpret_box_owner(self.DnB.box_owner)
        }

        return obs

    ## auxilary information. manhattan distance
    # action mask를 항상 포함한다.
    def _get_info(self):
        print(self.action_mask)
        return {
            'action_mask' : self.action_mask
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.DnB = DotsAndBoxes(self.n_box)

        # 불가능한 테두리의 행동들을 마스킹한다.
        def get_init_action_mask(n_box) -> np.ndarray:
            # shape: (n_box+1, n_box+1)
            r = np.arange(n_box + 1)[:, None]        # (R,1)
            c = np.arange(n_box + 1)[None, :]        # (1,C)

            mask = np.zeros((2, n_box + 1, n_box + 1), dtype=bool)
            # Horizontal mask: True only when c == n_box
            
            mask[0] = (c == n_box)
            # Vertical mask: True only when r == n_box
            mask[1] = (r == n_box)
            # Stack so that mask[0] = H, mask[1] = V
            
            return mask
        self.action_mask = get_init_action_mask(self.n_box)


        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    # render_mode가 rgb_array인 경우에만 np.ndarray 반환
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        
    # frame을 render한다.
    # render_mode가 'human'인 경우에는 pygame 창에 그린다.
    def _render_frame(self):
        if self.screen is None and self.clock is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen, self.clock, self.fonts = get_render_desc(self.n_box)

        if self.render_mode == "human":
            draw_board(self.screen, self.DnB, self.fonts)
            pygame.event.pump()
            pygame.display.update()
            
            self.clock.tick(self.metadata['render_fps'])

        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        
    # 행동 수행
    def step(self, action):
        assert not self.action_mask[action[0], action[1], action[2]], "The action is masked. choose another action."

        i_action = interpret_action(action)
        self.action_mask[action[0], action[1], action[2]] = True
        self.DnB.claim_edge(i_action)
        
        terminated = self.DnB.is_game_over
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
    

def main():
    n_box = 5
    env = DnBEnv(render_mode='human', n_box=n_box)

    observation, info = env.reset()
    action_mask = info['action_mask']
    print("action mask:", action_mask)

    print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0

    while not episode_over:

        action = env.action_space.sample()

        while action_mask[action[0], action[1], action[2]]:
            action = env.action_space.sample()
        
        print('Number of Claimed Edges:', np.sum(action_mask == False))

        observation, reward, terminated, truncated, info = env.step(action)
        action_mask = info['action_mask']
                
        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    print(f'Action spasce: {env.action_space}')
    print(f'Observation spasce: {env.observation_space}')

    env.close()


if __name__ == '__main__':
    main()