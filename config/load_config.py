import os
from typing import Dict, Any

import json

from policy import RandomPolicy, FixedOrderPolicy, SearchPolicy, OpeningPolicy, MixedPolicy
from search import AB_SearchEngine
from util.scheduler import ConstantScheduler, ExponentialSchedulerInt, PiecewiseConstantScheduler
from util.budget_manager import BudgetManager_v1, BudgetManager_v2, BudgetManager_v3
from heuristic import move_ordering, move_ordering_v2, evaluate_rel, evaluate_relv2, evaluate_relv3, evaluate_chain_aware, evaluate_comps

def load_config(path: str) -> dict[str, Any]:

    with open(path, mode='r') as f:
        config_json = json.load(f)
    
    run_name = config_json["run_name"]
    n_episodes = config_json["n_episodes"]
    p0_policy = make_obejct_from_config(config_json['p0_policy'])
    p1_policy = make_obejct_from_config(config_json['p1_policy'])
    
    run_name = config_json["run_name"]
    return run_name, n_episodes, p0_policy, p1_policy, config_json

POLICY_MAP = {
    'RandomPolicy': RandomPolicy,
    'FixedOrderPolicy': FixedOrderPolicy,
    'SearchPolicy': SearchPolicy,
    'OpeningPolicy': OpeningPolicy,
    'MixedPolicy': MixedPolicy
}

SEARCH_MAP = {
    'AB_SearchEngine': AB_SearchEngine
}

SCHEDULER_MAP = {
    'ConstantScheduler': ConstantScheduler, 
    'ExponentialSchedulerInt': ExponentialSchedulerInt, 
    'PiecewiseConstantScheduler': PiecewiseConstantScheduler
}

BUDGETMANAGER_MAP = {
    'BudgetManager_v1': BudgetManager_v1, 
    'BudgetManager_v2': BudgetManager_v2, 
    'BudgetManager_v3': BudgetManager_v3
}

FUNCTION_MAP = {
    'move_ordering': move_ordering, 
    'move_ordering_v2': move_ordering_v2, 
    'evaluate_rel': evaluate_rel, 
    'evaluate_relv2': evaluate_relv2, 
    'evaluate_relv3': evaluate_relv3, 
    'evaluate_chain_aware': evaluate_chain_aware, 
    'evaluate_comps': evaluate_comps
}

def make_obejct_from_config(config: dict) -> dict[str, Any]:
    MAPS = [POLICY_MAP, SEARCH_MAP, SCHEDULER_MAP, BUDGETMANAGER_MAP]
    param = {}
    obj_class = None

    for k, v in config.items():
        
        if k == 'type':
            object_type = config[k]
            for MAP in MAPS:
                obj_class = MAP.get(object_type)
                if obj_class != None:
                    break
            continue

        if isinstance(v, str) and FUNCTION_MAP.get(v) != None:
            param[k] = FUNCTION_MAP[v]
        elif isinstance(v, dict):
            param[k] = make_obejct_from_config(v)
        else:
            param[k] = v

    if obj_class == None:   
        return param
    return obj_class(**param)
