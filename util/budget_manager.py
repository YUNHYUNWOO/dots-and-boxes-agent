import numpy as np
from scipy.stats import skewnorm

from config import *
from .time_manager import TimeManager
from .scheduler import BaseScheduler, BudgetScheduler_v2, BudgetScheduler_v3

class BaseBudgetManager():
    def __init__(self):
        pass

    def get_budget(self, t: int, time_manager: TimeManager) -> float:
        pass


class BudgetManager_v1(BaseBudgetManager):
    def __init__(self, w_scheduler: BaseScheduler, MIN_BUDGET=0.02, MAX_BUDGET=float('inf'), SAFETY=0.05):
        self.w_scheduler = w_scheduler
        self.MIN_BUDGET = MIN_BUDGET
        self.MAX_BUDGET = MAX_BUDGET
        self.SAFETY = SAFETY

    def get_budget(self, t: int, time_manager: TimeManager) -> float:
        """
            budget_for_this_move(t):
            -> returns budget for turn t
        """
        rem = time_manager.remaining()
        base = rem / ( 2 * N_BOX * (N_BOX + 1) - t)

        w = self.w_scheduler(t)
        budget = base * w

        MAX_BUDGET = max(rem, self.MAX_BUDGET)    # 남은 시간 이상은 쓸 수 없음
        budget = max(self.MIN_BUDGET, min(budget, MAX_BUDGET))
        budget = max(0.0, budget - self.SAFETY)

        return budget
    

class BudgetManager_v2(BaseBudgetManager):
    def __init__(self, 
                 center: float,
                 scale: float,
                 alpha: float,
                 p: float, 
                 MIN_BUDGET=0.02, 
                 MAX_BUDGET=float('inf'), 
                 SAFETY=0.05):
        """
            center=33, scale=7, alpha=0, p=0.5 is good point to start
        """
        num_turns = H_COUNT + V_COUNT

        t = np.arange(num_turns)
        g = skewnorm.pdf(t, alpha, loc=center, scale=scale)
        w = g / g.sum()
        u = np.ones((num_turns,)) / num_turns
        self.w = np.cumsum(p * u + w * (1 - p))

        self.MIN_BUDGET = MIN_BUDGET
        self.MAX_BUDGET = MAX_BUDGET
        self.SAFETY = SAFETY

    def get_budget(self, t: int, time_manager: TimeManager):
        """
            get_budget(self, t: int, time_manager: TimeManager):
            -> returns budget for turn t
        """
        target_frac = self.w[t]

        rem = time_manager.remaining()
        used_frac = 1 - rem / time_manager.total_budget

        delta_frac = target_frac - used_frac

        # 타겟 프랙션에 맞는 '이론상' 예산
        budget = time_manager.total_budget * delta_frac

        MAX_BUDGET = min(rem, self.MAX_BUDGET)    # 남은 시간 이상은 쓸 수 없음
        budget = max(self.MIN_BUDGET, min(budget, MAX_BUDGET))
        budget = max(0.0, budget - self.SAFETY)
        
        return budget
    

class BudgetManager_v3(BaseBudgetManager):
    def __init__(self, 
                 center: float,
                 scale: float,
                 alpha: float,
                 p: float,
                 w_2:float, 
                 MIN_BUDGET=0.02, 
                 MAX_BUDGET=float('inf'), 
                 SAFETY=0.05):
        """
        center=30, scale=7, alpha=1, p=0.3, w_2=2.0
        """
        
        num_turns = H_COUNT + V_COUNT
        t = np.arange(num_turns)
        g = skewnorm.pdf(t, alpha, loc=center, scale=scale)
        
        w = g / g.sum()
        u = np.ones((num_turns,)) / num_turns
        self.w = (p * u + w * (1 - p)) * w_2
    
        self.MIN_BUDGET = MIN_BUDGET
        self.MAX_BUDGET = MAX_BUDGET
        self.SAFETY = SAFETY

    def get_budget(self, t: int, time_manager: TimeManager):
        """
            get_budget(self, t: int, time_manager: TimeManager):
            -> returns budget for turn t
        """
        rem = time_manager.remaining()
        w = self.w[t]

        # 타겟 프랙션에 맞는 '이론상' 예산
        budget = time_manager.total_budget * w

        MAX_BUDGET = min(rem, self.MAX_BUDGET)    # 남은 시간 이상은 쓸 수 없음
        budget = max(self.MIN_BUDGET, min(budget, MAX_BUDGET))
        budget = max(0.0, budget - self.SAFETY)
        
        return budget