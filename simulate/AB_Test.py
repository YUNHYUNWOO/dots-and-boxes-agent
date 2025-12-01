"""Simple A/B test runner between two policies."""
from dotsandboxes import DnBEnv
from policy import BasePolicy
from util.time_manager import TimeManager


def Test_AB_Model(env: DnBEnv, A_policy: BasePolicy, B_policy: BasePolicy, verbose: bool = False) -> None:
    """Play a single game between two policies and print debug info."""

    observation, info = env.reset()

    if verbose:
        print(f"Starting observation: {observation}")

    episode_over = False
    A_time_manager = TimeManager()
    B_time_manager = TimeManager()

    while not episode_over:
        A_time_manager.start_move()
        A_action = A_policy.get_action(observation, A_time_manager)
        A_time_manager.end_move()

        B_time_manager.start_move()
        B_action = B_policy.get_action(observation, A_time_manager)
        B_time_manager.end_move()

        print(f"A_action: {A_action}, B_action: {B_action}")
        print(f"A_log: {A_policy.get_log()}, B_log: {B_policy.get_log()}")

        observation, _, terminated, truncated, info = env.step(A_action)
        episode_over = terminated or truncated
    env.close()

if __name__ is "__main__":
    from util import BudgetManager_pdf_base, ExponentialSchedulerInt
    from policy import SearchPolicy
    from heuristic import evaluate_bad_moves, move_ordering
    from search import AB_SearchEngine

    env = DnBEnv('human')
    p0_config = {
        "evaluate": evaluate_bad_moves,
        "move_ordering": move_ordering,
        "depth": ExponentialSchedulerInt(15, 2, 35, 20),
        "use_iterative_deepening": True,
        "deterministic": True,
        "skip_move": False,
        "use_extension": False,
        "use_time_control": False,
        "budget_manager": BudgetManager_pdf_base(30, 7, 3, 0.3, 2.0),
        "use_pvs": False
    }
    p0_policy = SearchPolicy(AB_SearchEngine(), p0_config)

    p1_config = {
        "evaluate": evaluate_bad_moves,
        "move_ordering": move_ordering,
        "depth": ExponentialSchedulerInt(15, 2, 35, 20),
        "use_iterative_deepening": True,
        "deterministic": True,
        "skip_move": False,
        "use_extension": False,
        "use_time_control": False,
        "budget_manager": BudgetManager_pdf_base(30, 7, 3, 0.3, 2.0),
        "use_pvs": False,
        "use_aspiration": True
    }
    p1_policy = SearchPolicy(AB_SearchEngine(), p1_config)

    Test_AB_Model(env, p0_policy, p1_policy)
    
