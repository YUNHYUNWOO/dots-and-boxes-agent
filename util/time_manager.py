"""Utilities for tracking per-move and total time budgets."""

import time
from typing import Optional


class TimeManager:
    """
    Track cumulative time usage and optional move deadlines.
    Standard usage is :
    1. get remaining with self.remaining()
    2. assign budget for remaining
    3. set deadlines with self.set_deadline(self, budget).
    4. self.check_time() will raise error when cur time exceed the deadline
    """

    def __init__(self) -> None:
        self.total_budget: float = 24.0
        self.used_time: float = 0.0
        self._move_start: Optional[float] = None
        self._deadline: Optional[float] = None

    def remaining(self) -> float:
        """Return the remaining total time budget."""

        return max(0.0, self.total_budget - self.used_time)

    def start_move(self) -> None:
        """Mark the start of a move."""

        self._move_start = time.perf_counter()

    def end_move(self) -> float:
        """Update used time based on the elapsed duration of the move."""

        if self._move_start is None:
            return 0.0
        dt = time.perf_counter() - self._move_start
        self.used_time += dt
        return dt

    def set_deadline(self, budget: float) -> None:
        """Set an absolute deadline ``budget`` seconds after ``start_move``."""

        base_time = self._move_start or time.perf_counter()
        self._deadline = base_time + budget

    def check_time(self) -> None:
        """Raise ``TimeoutError`` if the move deadline has been exceeded."""

        if self._deadline is None:
            return
        if time.perf_counter() >= self._deadline:
            raise TimeoutError()

    def get_move_start_time(self) -> Optional[float]:
        """Expose the timestamp recorded at :meth:`start_move`."""

        return self._move_start

    def reset(self) -> None:
        """Reset budgets and deadlines to their defaults."""

        self.total_budget = 24.0
        self.used_time = 0.0
        self._move_start = None
        self._deadline = None
