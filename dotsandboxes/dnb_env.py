"""Gymnasium environment wrapper for Dots and Boxes using (c, r, d) actions."""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from config import Action, Board, N, N_BOX
from dotsandboxes.dnb import DotsAndBoxes, draw_board, get_render_context


def to_env_board(h_edges: list[list], v_edges: list[list]) -> Board:
    """Convert engine edges (row-major) to env Board layout (col, row, dir)."""

    def to_bool(edge) -> bool:
        return edge is not None

    board: Board = [[[0 for _ in range(2)] for _ in range(N)] for _ in range(N)]

    for r in range(N):
        for c in range(N):
            if c != N - 1:
                board[c][r][0] = to_bool(h_edges[r][c])
            if r != N - 1:
                board[c][r][1] = to_bool(v_edges[r][c])
    return board


def from_env_action(action: Action) -> tuple[str, int, int]:
    """Map env action (c, r, d) to engine format (ori, r, c)."""

    ori = "H" if action[2] == 0 else "V"
    c, r = map(int, action[:2])
    return ori, r, c


def get_init_action_mask() -> np.ndarray:
    """Initial action mask that hides perimeter edges that do not exist."""

    r = np.arange(N_BOX + 1)[None, :]
    c = np.arange(N_BOX + 1)[:, None]
    mask = np.zeros((N_BOX + 1, N_BOX + 1, 2), dtype=bool)
    mask[:, :, 0] = c == N_BOX  # hide rightmost nonexistent horizontals
    mask[:, :, 1] = r == N_BOX  # hide bottom nonexistent verticals
    return mask


class DnBEnv(gym.Env):
    """Minimal Gymnasium wrapper around the DotsAndBoxes engine."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.DnB: DotsAndBoxes | None = None
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "edges": spaces.Box(0, 1, shape=(N, N, 2), dtype=bool),
                "cur_player": spaces.Discrete(2),
            }
        )

        # action: (c, r, d) with d in {0=H, 1=V}
        self.action_space = spaces.MultiDiscrete(nvec=(N, N, 2), dtype=int)
        self.action_mask: np.ndarray | None = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.screen = None
        self.clock = None
        self.fonts = None

    def _get_obs(self) -> dict:
        """Return the current observation dict."""

        return {
            "board": to_env_board(self.DnB.h_edges, self.DnB.v_edges),
            "cur_player": self.DnB.current_player,
            "score": self.DnB.score,
        }

    def _get_info(self) -> dict:
        """Return auxiliary info including action mask and score."""

        return {
            "winner": self.DnB.winner(),
            "action_mask": self.action_mask,
            "score": self.DnB.score,
        }

    def reset(self, seed=None, options=None) -> tuple[dict, dict]:
        super().reset(seed=seed)

        self.DnB = DotsAndBoxes(N_BOX)
        self.action_mask = get_init_action_mask()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def render(self) -> np.ndarray | None:
        """Render the current frame and return RGB array when requested."""

        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def _render_frame(self) -> np.ndarray | None:
        if self.screen is None and self.clock is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen, self.clock, self.fonts = get_render_context(N_BOX)

        if self.render_mode == "human":
            draw_board(self.screen, self.DnB, self.fonts)
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

    def step(self, action: Action) -> tuple[dict, int, bool, bool, dict]:
        """Apply an action and return (obs, reward, terminated, truncated, info)."""


        assert (
            not self.action_mask[action[0], action[1], action[2]]
        ), "The action has already been taken. Choose another action."

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
        """Tear down pygame resources."""

        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen, self.clock, self.fonts = None, None, None


def main():
    env = DnBEnv(render_mode="human")

    observation, info = env.reset()
    action_mask = info["action_mask"]

    print(f"Starting observation: {observation}")
    episode_over = False
    total_reward = 0

    while not episode_over:
        action = env.action_space.sample()
        while action_mask[action[0], action[1], action[2]]:
            action = env.action_space.sample()

        print("Number of Claimed Edges:", np.sum(action_mask == False))

        observation, reward, terminated, truncated, info = env.step(action)
        action_mask = info["action_mask"]
        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Winner: {info['winner']} Total reward: {total_reward}")
    print(f"observation: {observation}")
    print(f"info: {info}")

    env.close()


if __name__ == "__main__":
    main()
