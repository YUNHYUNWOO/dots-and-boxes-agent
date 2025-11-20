from enum import Enum

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from config import *

from .dnb import DotsAndBoxes, get_render_context, draw_board



def to_env_board(h_edges: list[list], v_edges: list[list]) -> Board:
    """
    DotsAndBoxes의 보드 상태를 DnB Env(대회의 좌표계)의 형태로 변환
    # r, c -> c, r / None -> 0, (-1,1) -> 1
    """
    def map_func(e):
        return (e != None)
    
    board = [[[0 for _ in range(2)] for _ in range(N)] for _ in range(N)]

    for i in range(N):
        for j in range(N):
            if j != N - 1: 
                board[j][i][0] = map_func(h_edges[i][j])
            if i != N - 1:
                board[j][i][1] = map_func(v_edges[i][j])
    return board


def from_env_action(action: Action) -> tuple[str, int, int]:
    """
    DnB Env(대회의 좌표계)의 action을 DotsAndBoxes의 형태로 변환
    c, r, d -> d(str), r, c
    """
    ori = 'H' if action[2] == 0 else 'V'
    c, r = map(int, action[:2])
    return (ori, r, c)

def get_init_action_mask() -> np.ndarray:
    # shape: (n_box+1, n_box+1)
    r = np.arange(N_BOX + 1)[None, :]        # (R,1)
    c = np.arange(N_BOX + 1)[:, None]        # (1,C)
    mask = np.zeros((N_BOX + 1, N_BOX + 1, 2), dtype=bool)
    # Horizontal mask: True only when c == n_box
    
    mask[:,:,0] = (c == N_BOX)
    # Vertical mask: True only when r == n_box
    mask[:,:,1] = (r == N_BOX)
    # Stack so that mask[0] = H, mask[1] = V
    return mask

class DnBEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.DnB = None # Game 인스턴스, 초기화는 reset에서 수행
        self.window_size = 512 #The size of the Pygame window

        # Observation space 정의
        # edges: [r, c, z(방향)] boolean 배열
        # cur_player: 현재 플레이어 (0: 유저1, 1: 유저2)

        self.observation_space = spaces.Dict(
            {
                "edges": spaces.Box(0, 1, shape=(N,N,2), dtype=bool),
                "cur_player": spaces.Discrete(2)
            }
        )

        # action space 정의
        # action: (edge_type, row, col)
        # edge_type: 0 (h_edge), 1 (v_edge)
        # row: 0~5 (h_edge), 0~4 (v_edge) 
        # col: 0~4 (h_edge), 0~5 (v_edge)
        # 불가능한 행동은 action mask로 후에 처리z
        self.action_space = spaces.MultiDiscrete(nvec=(N,N,2), dtype=int)
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


    def _get_obs(self) -> dict:

        obs = {
            'board': to_env_board(self.DnB.h_edges, self.DnB.v_edges),
            'cur_player': self.DnB.current_player,
            'score' : self.DnB.score,
        }

        return obs

    # auxilary information.
    # action mask를 항상 포함한다.
    def _get_info(self) -> dict:
        
        return {
            'winner': self.DnB.winner(),
            'action_mask' : self.action_mask,
            'score' : self.DnB.score,
        }
    
    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)

        self.DnB = DotsAndBoxes(N_BOX)

        # 불가능한 테두리의 행동들을 마스킹한다.

        self.action_mask = get_init_action_mask()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    # render_mode가 rgb_array인 경우에만 np.ndarray 반환
    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    # frame을 render한다.
    # render_mode가 'human'인 경우에는 pygame 창에 그린다.
    def _render_frame(self) -> np.ndarray | None:
        if self.screen is None and self.clock is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.screen, self.clock, self.fonts = get_render_context(N_BOX)

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
    def step(self, action:Action) -> tuple[dict, int, bool, bool, dict]:
        assert not self.action_mask[action[0], action[1], action[2]], "The action has already been taken. Choose another action."

        i_action = from_env_action(action)
        self.action_mask[action[0], action[1], action[2]] = True
        self.DnB.claim_edge(i_action)
        
        terminated = self.DnB.is_game_over
        reward = 1 if terminated else 0
        if terminated:
            reward *= -1 if (self.DnB.winner() != self.DnB.current_player) else 1
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
    env = DnBEnv(render_mode='human')

    observation, info = env.reset()
    action_mask = info['action_mask']

    print(f"Starting observation: {observation}")
    #print(len(observation['edges']), len(observation['edges'][0]), len(observation['edges'][0][0]))
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

    print(f"Episode finished! Winner: {info['winner']} Total reward: {total_reward}")
    print(f'observation: {observation}')
    print(f'info: {info}')

    env.close()


if __name__ == '__main__':
    main()