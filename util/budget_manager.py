"""Budget managers that allocate per-move time based on schedulers."""

import numpy as np
from scipy.stats import skewnorm

from config import H_COUNT, N_BOX, V_COUNT
from util.scheduler import BaseScheduler
from util.time_manager import TimeManager


class BaseBudgetManager:
    """Interface for computing a time budget for a move."""

    def get_budget(self, t: int, time_manager: TimeManager) -> float:
        """Return a time budget for ply ``t``."""

        raise NotImplementedError


class BudgetManager_default(BaseBudgetManager):
    """Simple scheduler-scaled budget with a safety margin."""

    def __init__(
        self,
        w_scheduler: BaseScheduler,
        MIN_BUDGET: float = 0.02,
        MAX_BUDGET: float = float("inf"),
        SAFETY: float = 0.05,
    ) -> None:
        self.w_scheduler = w_scheduler
        self.MIN_BUDGET = MIN_BUDGET
        self.MAX_BUDGET = MAX_BUDGET
        self.SAFETY = SAFETY

    def get_budget(self, t: int, time_manager: TimeManager) -> float:
        """Allocate budget proportional to remaining moves."""

        rem = time_manager.remaining()
        base = rem / (2 * N_BOX * (N_BOX + 1) - t)

        weight = self.w_scheduler(t)
        budget = base * weight

        max_budget = min(rem, self.MAX_BUDGET)
        budget = max(self.MIN_BUDGET, min(budget, max_budget))
        budget = max(0.0, budget - self.SAFETY)

        return budget


class BudgetManager_cdf_base(BaseBudgetManager):
    """Skew-normal cumulative schedule targeting specific plies."""

    def __init__(
        self,
        center: float,
        scale: float,
        alpha: float,
        p: float,
        MIN_BUDGET: float = 0.02,
        MAX_BUDGET: float = float("inf"),
        SAFETY: float = 0.05,
    ) -> None:
        self.MIN_BUDGET = MIN_BUDGET
        self.MAX_BUDGET = MAX_BUDGET
        self.SAFETY = SAFETY

        num_turns = H_COUNT + V_COUNT

        t = np.arange(num_turns)
        g = skewnorm.pdf(t, alpha, loc=center, scale=scale)
        w = g / g.sum()
        uniform = np.ones((num_turns,)) / num_turns
        self.w = np.cumsum(p * uniform + w * (1 - p))

    def get_budget(self, t: int, time_manager: TimeManager) -> float:
        """Allocate budget to track target cumulative fraction."""

        target_frac = self.w[t]

        rem = time_manager.remaining()
        used_frac = 1 - rem / time_manager.total_budget

        delta_frac = target_frac - used_frac
        budget = time_manager.total_budget * delta_frac

        max_budget = min(rem, self.MAX_BUDGET)
        budget = max(self.MIN_BUDGET, min(budget, max_budget))
        budget = max(0.0, budget - self.SAFETY)

        return budget


class BudgetManager_pdf_base(BaseBudgetManager):
    """Scaled skew-normal schedule for larger midgame emphasis."""

    def __init__(
        self,
        center: float,
        scale: float,
        alpha: float,
        p: float,
        w_2: float,
        MIN_BUDGET: float = 0.02,
        MAX_BUDGET: float = float("inf"),
        SAFETY: float = 0.05,
    ) -> None:
        self.MIN_BUDGET = MIN_BUDGET
        self.MAX_BUDGET = MAX_BUDGET
        self.SAFETY = SAFETY

        num_turns = H_COUNT + V_COUNT
        t = np.arange(num_turns)
        g = skewnorm.pdf(t, alpha, loc=center, scale=scale)

        w = g / g.sum()
        uniform = np.ones((num_turns,)) / num_turns
        self.w = (p * uniform + w * (1 - p)) * w_2

    def get_budget(self, t: int, time_manager: TimeManager) -> float:
        """Allocate budget following the scaled skew-normal profile."""

        rem = time_manager.remaining()
        weight = self.w[t]

        budget = time_manager.total_budget * weight

        max_budget = min(rem, self.MAX_BUDGET)
        budget = max(self.MIN_BUDGET, min(budget, max_budget))
        budget = max(0.0, budget - self.SAFETY)

        return budget
