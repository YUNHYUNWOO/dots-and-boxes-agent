"""Policy that delegates to other policies based on a schedule."""

from typing import Any, Dict

from config import Action, Board, N

from .basepolicy import BasePolicy
from util.scheduler import PiecewiseConstantScheduler
from util.time_manager import TimeManager


class MixedPolicy(BasePolicy):
    """Select among multiple policies according to a time-dependent schedule."""

    def __init__(self, policy_scheduler: PiecewiseConstantScheduler):
        super().__init__()
        self.policy_scheduler = policy_scheduler

    def get_policy(self, t: int):
        """Return the policy scheduled for ply ``t``."""

        return self.policy_scheduler.value(t)

    def get_action(self, observation: Dict[str, Any], time_manager: TimeManager) -> Action:
        """Delegate action selection to the policy chosen for the current ply."""

        def get_t(board: Board) -> int:
            t = 0
            for c in range(N):
                for r in range(N):
                    for d in range(2):
                        if board[c][r][d]:
                            t += 1
            return t

        t = get_t(observation['board'])

        policy = self.get_policy(t)

        return policy.get_action(observation, time_manager)
