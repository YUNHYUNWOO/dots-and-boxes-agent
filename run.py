
import os
import time

import numpy as np
import tqdm.auto as tqdm

from dotsandboxes import DnBEnv
from policy import BasePolicy, TimeManager, OpeningPolicy, FixedOrderPolicy, SearchPolicy
from util import *
from heuristic import evaluate_rel, evaluate_comps, move_ordering
from search import AB_SearchEngine
from util.scheduler import ExponentialSchedulerInt, BooleanScheduler
from util.budget_manager import BudgetManager_v1, BudgetManager_v2, BudgetManager_v3

from simulate import simulate_episode, simulate_multiple_episodes



def main():
    BASE_SAVE_PATH = 'simulate/sim_result'
    env = DnBEnv(render_mode='human')

    config_p0 = {
        'evaluate':evaluate_rel,
        'move_ordering':move_ordering,
        'depth': ExponentialSchedulerInt(15,2,35,24),
        'use_iterative_deepening': True,
        'deterministic': BooleanScheduler(true_intervals=[[10, 60]], default=False),
        'skip_move': False,
        'use_extension': False,
        'use_time_control': True,
        'budget_manager': BudgetManager_v1(ExponentialScheduler(15,2,20,4.2)),
        'use_pvs': True
    }
    p0_policy = SearchPolicy(AB_SearchEngine(), config_p0)

    
    config_p1 = {
        'evaluate':evaluate_rel,
        'move_ordering':move_ordering,
        'depth': 30,
        'use_iterative_deepening': True,
        'deterministic': BooleanScheduler(true_intervals=[[10, 60]], default=False),
        'skip_move': False,
        'use_extension': False,
        'use_time_control': True,
        'budget_manager': BudgetManager_v3(center=26, scale=7, alpha=1, p=0.3, w_2=2.0),
        'use_pvs': True
    }
    p1_policy = SearchPolicy(AB_SearchEngine(), config_p1)

    run_name = 'test_2'
    env.render_mode = 'rgb_array'
    save_path = os.path.join(BASE_SAVE_PATH, run_name)
    simulate_multiple_episodes(env, p0_policy, p1_policy, n_episodes=4, log=True, save_path=save_path, verbose=False)

if __name__ == '__main__':
    main()