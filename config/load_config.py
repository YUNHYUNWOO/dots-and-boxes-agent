"""Helpers for instantiating policies and schedulers from JSON configs."""

import json
from typing import Any, Dict, Tuple, Optional

from heuristic import (
    evaluate_chain,
    evaluate_default,
    evaluate_bad_moves,
    move_ordering,
    move_ordering_v2,
    complete_extension,
    give_away_extension
)
from policy import BasePolicy, FixedOrderPolicy, MixedPolicy, OpeningPolicy, RandomPolicy, SearchPolicy
from search import AB_SearchEngine
from util.budget_manager import BudgetManager_default, BudgetManager_cdf_base, BudgetManager_pdf_base
from util.scheduler import (
    BooleanScheduler,
    ConstantScheduler,
    CosineRestartScheduler,
    CosineScheduler,
    CyclicalScheduler,
    ExponentialScheduler,
    ExponentialSchedulerInt,
    InverseSqrtScheduler,
    LinearScheduler,
    LinearSchedulerInt,
    PiecewiseConstantScheduler,
    PiecewiseScheduler,
    PolynomialScheduler,
    SigmoidScheduler,
    StepScheduler,
    WarmupHoldDecayScheduler,
)

def load_config(path: str) -> Tuple[str, int, BasePolicy, BasePolicy, Dict[str, Any]]:
    """Load a configuration file and instantiate the described policies."""

    with open(path, mode="r", encoding="utf-8") as f:
        config_json = json.load(f)

    run_name = config_json["run_name"]
    n_episodes = config_json["n_episodes"]
    p0_policy = make_object_from_config(config_json["p0_policy"])
    p1_policy = make_object_from_config(config_json["p1_policy"])

    return run_name, n_episodes, p0_policy, p1_policy, config_json


POLICY_MAP = {
    "RandomPolicy": RandomPolicy,
    "FixedOrderPolicy": FixedOrderPolicy,
    "SearchPolicy": SearchPolicy,
    "OpeningPolicy": OpeningPolicy,
    "MixedPolicy": MixedPolicy,
}

SEARCH_MAP = {"AB_SearchEngine": AB_SearchEngine}

SCHEDULER_MAP = {
    "BooleanScheduler": BooleanScheduler,
    "ConstantScheduler": ConstantScheduler,
    "CosineRestartScheduler": CosineRestartScheduler,
    "CosineScheduler": CosineScheduler,
    "CyclicalScheduler": CyclicalScheduler,
    "ExponentialScheduler": ExponentialScheduler,
    "ExponentialSchedulerInt": ExponentialSchedulerInt,
    "InverseSqrtScheduler": InverseSqrtScheduler,
    "LinearScheduler": LinearScheduler,
    "LinearSchedulerInt": LinearSchedulerInt,
    "PiecewiseConstantScheduler": PiecewiseConstantScheduler,
    "PiecewiseScheduler": PiecewiseScheduler,
    "PolynomialScheduler": PolynomialScheduler,
    "SigmoidScheduler": SigmoidScheduler,
    "StepScheduler": StepScheduler,
    "WarmupHoldDecayScheduler": WarmupHoldDecayScheduler,
}

BUDGETMANAGER_MAP = {
    "BudgetManager_default": BudgetManager_default,
    "BudgetManager_cdf_base": BudgetManager_cdf_base,
    "BudgetManager_pdf_base": BudgetManager_pdf_base,
}

FUNCTION_MAP = {
    "move_ordering": move_ordering,
    "move_ordering_v2": move_ordering_v2,
    "evaluate_default": evaluate_default,
    "evaluate_bad_moves": evaluate_bad_moves,
    "evaluate_chain": evaluate_chain,
    "complete_extension": complete_extension,
    "give_away_extensioin": give_away_extension
}


def make_object_from_config(config: dict | str) -> Dict[str, Any]:
    """Recursively construct an object or parameter dict from a JSON blob."""

    maps = [POLICY_MAP, SEARCH_MAP, SCHEDULER_MAP, BUDGETMANAGER_MAP, FUNCTION_MAP]
    param: Dict[str, Any] = {}
    obj_class = None

    if isinstance(config, str):
        for mapping in maps:
            obj_class = mapping.get(config)
            if obj_class is not None:
                return obj_class

    for key, value in config.items():
        if key == "type":
            object_type = config[key]
            for mapping in maps:
                obj_class = mapping.get(object_type)
                if obj_class is not None:
                    break
            continue

        if isinstance(value, str) and FUNCTION_MAP.get(value) is not None:
            param[key] = make_obejct_from_config(value)
        elif isinstance(value, dict):
            param[key] = make_object_from_config(value)
        elif isinstance(value, list):
            param[key] = [make_obejct_from_config(v) for v in value]
        else:
            param[key] = value

    if obj_class is None:
        return param
    return obj_class(**param)


# Backwards compatibility for legacy imports.
make_obejct_from_config = make_object_from_config
