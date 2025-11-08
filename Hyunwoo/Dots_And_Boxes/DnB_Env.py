from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from DnB import DotsAndBoxes, get_render_desc, draw_board

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

# 여기선는 유저를 -1, 1로 구분
# DnB에서는 0, 1로 구분
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
        self.DnB = None
        self.window_size = 512 #The size of the Pygame window

        self.observation_space = spaces.Dict(
        {
            "h_edges": spaces.Box(-1, 1, shape=(6,5), dtype=int),
            "v_edges": spaces.Box(-1, 1, shape=(5,6), dtype=int),
            "box_owner": spaces.Box(-1, 1, shape=(5,5), dtype=int)
        }
    )
        
        self.h_board = np.zeros(shape=(6,5), dtype=bool)
        self.v_board = np.zeros(shape=(5,6), dtype=bool)

        self.action_space = spaces.MultiDiscrete(nvec=[2,6,6], dtype=int)


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
    def _get_info(self):
        return {
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.DnB = DotsAndBoxes(self.n_box)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

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
        
    def step(self, action):
        
        i_action = interpret_action(action)
        self.DnB.claim_edge(i_action)
        
        terminated = self.DnB.is_game_over
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    

def get_action_sample_mask(n_box):
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

def main():
    n_box = 5
    env = DnBEnv(render_mode='human', n_box=n_box)
    mask = get_action_sample_mask(n_box)

    observation, info = env.reset()

    print(f"Starting observation: {observation}")

    def update_action_mask(observation):    
        for r in range(len(observation['h_edges'])):
            for c, e in enumerate(observation['h_edges'][r]):
                mask[0, r, c] |= e
        for r in range(len(observation['v_edges'])):
            for c, e in enumerate(observation['v_edges'][r]):
                mask[1, r, c] |= e
        
    episode_over = False
    total_reward = 0

    while not episode_over:

        action = env.action_space.sample()
        while mask[*action]:
            action = env.action_space.sample()
            
        print('action:', action)
        print('Number of Claimed Edges:', np.sum(mask == False))

        observation, reward, terminated, truncated, info = env.step(action)
        
        update_action_mask(observation)
                
        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    print(f'Action spasce: {env.action_space}')
    print(f'Observation spasce: {env.observation_space}')

    env.close()


if __name__ == '__main__':
    main()