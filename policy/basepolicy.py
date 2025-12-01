"""Policy implementations for selecting moves in Dots and Boxes."""

import random
from typing import Any, Dict

from config import Action, H, N_BOX, V
from util.time_manager import TimeManager


class BasePolicy:
    """Base interface for all policies.

    Subclasses should implement :meth:`get_action` to return a valid action
    using the provided observation and time budget.
    """

    def __init__(self) -> None:
        # Each policy shares a TimeManager instance to coordinate time budgets.
        self.time_manager = TimeManager()

    def get_action(self, observation: Dict[str, Any], time_manager: TimeManager) -> Action:
        """Return the next action given the current observation."""

        # The observation contains the agent's visible board state and metadata.
        # An action mask indicating valid moves is always included.
        raise NotImplementedError

    def get_log(self):
        """Return optional debug information collected during decision making."""

        return None


class RandomPolicy(BasePolicy):
    """Sample uniformly from legal moves, respecting the action mask."""

    def __init__(self) -> None:
        super().__init__()

    def get_action(self, observation: Dict[str, Any], time_manager: TimeManager) -> Action:
        board = observation["board"]

        def sample_action() -> Action:
            c = random.randrange(0, N_BOX + 1)
            r = random.randrange(0, N_BOX + 1)
            d = random.randrange(0, 2)
            if (d == H and c == N_BOX) or (d == V and r == N_BOX):
                return sample_action()
            return (c, r, d)

        action = sample_action()
    
        while board[action[0]][action[1]][action[2]]:
            action = sample_action()

        return action


class FixedOrderPolicy(BasePolicy):
    """Iterate through a deterministic ordering of edges."""

    def __init__(self) -> None:
        super().__init__()
        self.action_order = []
        for ori in range(2):
            for r in range(N_BOX + 1):
                for c in range(N_BOX + 1):
                    if (ori == 0 and c == N_BOX) or (ori == 1 and r == N_BOX):
                        continue
                    self.action_order.append((c, r, ori))
        self.cur_index = 0

    def get_action(self, observation: Dict[str, Any], time_manager: TimeManager) -> Action:
        """Return the next unused edge in the precomputed order."""

        action = self.action_order[self.cur_index]

        while observation["board"][action[0]][action[1]][action[2]]:
            self.cur_index = (self.cur_index + 1) % len(self.action_order)
            action = self.action_order[self.cur_index]

        self.cur_index = (self.cur_index + 1) % len(self.action_order)

        return action
