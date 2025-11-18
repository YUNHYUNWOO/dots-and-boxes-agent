import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from DotsAndBoxes import DnBEnv
from Policy import *
from Search import *
from Util import *
import tqdm.auto as tqdm
import os
import time
from .Log import save_sim_logs

BASE_SAVE_PATH = './Simulate/SimResult'


# 정책에 의한 전체 에피소드를 시뮬레이션 하는 함수
# 만약 pygame window가 작동하지 않으면 env 생성시 render_mode를 'human'으로 설정할 것
def SimulateEpisode(env, p0_policy: BasePolicy, p1_policy: BasePolicy, verbose=False):
    """
    po_policy와 p1_policy가 선후공을 맡아 한 에피소드를 시뮬레이션합니다.

    Returns:
        results = {
            'record': List of action records for each episode (Submit format),
            'info': List of info dictionaries for each episode (env의 정의 그대로),
            'total_reward': List of total rewards for the first player in each episode (현재까지는 아무 쓸모 없음, Debugging용, Reward 알고리즘이 복잡하면 쓸모있을 수도)
        }
    """
        
    # verbose는 디버깅 출력 여부
    observation, info = env.reset()

    if verbose:
        print(f"Starting observation: {observation}")

    turn_count = [0, 0]
    time_spent = []
    Action_log = []
    vals = []
    scores = []
    player = []
    Policy_log = []
    episode_over = False
    p0_time_manager = TimeManager()
    p1_time_manager = TimeManager()

    while not episode_over:
        cur_player =  observation['cur_player']
        time_manager = p0_time_manager if cur_player == 0 else p1_time_manager
        policy = p0_policy if cur_player == 0 else p1_policy

        print(cur_player)
        t0 = time.perf_counter()
        time_manager.start_move()
        action, val = policy.get_action(observation, info, env, time_manager)
        time_manager.end_move()        
        t1 = time.perf_counter()
        Policy_log.append(policy.get_log())
        Policy_log[-1]['time_spent'] = t1 - t0

        if verbose:
            print('action:', action)
            print('action_mask:')
            for i in range(2):
                ori = 'H' if i == 0 else 'V'
                print(f'--{ori}--')
                for r in range(info['action_mask'].shape[0]):
                    print(info['action_mask'][:,r,i])
                print('-----')
            print('Number of Claimed Edges:', np.sum(info['action_mask'] == False))

        observation, _, terminated, truncated, info = env.step(action)
        episode_over = terminated or truncated
        
        player.append(cur_player)
        time_spent.append(t1 - t0)
        vals.append(val)
        scores.append([s for s in info['score']])

        Action_log.append([cur_player] + action)

    if verbose:
        print(f"Episode finished! Winner: {info['winner']}")
        print(f'Action spasce: {env.action_space}')
        print(f'Observation spasce: {env.observation_space}')

    env.close()
   
    Evaluation_log = {
        'player': player,
        'time_spent': time_spent,
        'vals': vals,
        'scores': scores,
        'winner': info['winner'],
    }
    Action_log = {'Action_log': '5, 5, ' + (", ".join(str(x) for a in Action_log for x in a))}
    Policy_log = {
        'player': player,
        'logs' : Policy_log
    }
    return Evaluation_log, Action_log, Policy_log


def SimulateMultipleEpisodes(env, p0_policy: BasePolicy, p1_policy: BasePolicy, n_episodes: int, verbose=False):
    """
    po_policy와 p1_policy가 번갈아가며 선후공을 맡아 n_episodes만큼 시뮬레이션을 수행합니다.
    결과는 딕셔너리 형태로 반환됩니다.

    Returns:
        results = {
            'record': List of action records for each episode (Submit format),
            'info': List of info dictionaries for each episode (env의 정의 + first_player),
            'total_reward': List of total rewards for the first player in each episode (현재까지는 아무 쓸모 없음, Debugging용, Reward 알고리즘이 복잡하면 쓸모있을 수도)
        }
    """

    results = {
        'record': [],
        'info': [],
        'total_reward': [],
        'time_spent': []
    }

    first_player = p0_policy
    second_player = p1_policy

    Evaluation_logs = []
    Action_logs = []
    Policy_logs = []

    for episode in tqdm.tqdm(range(n_episodes)):
        if verbose:
            print(f"=== Episode {episode + 1} ===")
        Evaluation_log, Action_log, Policy_log  = SimulateEpisode(env, first_player, second_player, verbose)

        Evaluation_log['first_player'] = 0 if first_player == p0_policy else 1

        if first_player == p1_policy:
            Evaluation_log['player'] = [1 - p for p in Evaluation_log['player']]
            Evaluation_log['winner'] = 1 - Evaluation_log['winner']
            Evaluation_log['scores'] = [score[::-1] for score in Evaluation_log['scores']]
            Policy_log['player'] = [1 - p for p in Policy_log['player']]
        Action_log['first_player'] = 0 if first_player == p0_policy else 1

        Evaluation_logs.append({
            'episode_id': episode,
            'Evaluation_log': Evaluation_log
        })

        Action_logs.append({
            'episode_id': episode,
            'Action_log': Action_log
        })

        Policy_logs.append({
            'episode_id': episode,
            'Policy_log': Policy_log
        })

        first_player, second_player = second_player, first_player  # 다음 에피소드에서 선후공 교체
        
    return Evaluation_logs, Action_logs, Policy_logs



if __name__ == "__main__":

    run_name = 'v1 vs v2'
    n_box = 5
    env = DnBEnv(render_mode='human', n_box=n_box)

    # p1_policy_part1 = OpeningPolicy()
    # config_p1 = {
    #     'evaluate':evaluate_rel,
    #     'move_ordering':None,
    #     'depth': ExponentialSchedulerInt(15, 2, 35, 5),
    #     'use_iterative_deepening': True,
    #     'deterministic': BooleanScheduler(true_intervals=[[10, 60]], default=False)
    # }
    # p1_policy_part2 = SearchPolicy(AB_TT_Search(), config_p1)
    # p1_policy_scheduler = PiecewiseConstantScheduler([[30, 60, p1_policy_part2]], default_value=p1_policy_part1)
    # p1_policy = MixedPolicy(p1_policy_scheduler)
    
    config_p0 = {
        'evaluate':evaluate_rel,
        'move_ordering':move_ordering,
        'depth': ExponentialSchedulerInt(15, 2, 35, 18),
        'use_iterative_deepening': True,
        'deterministic': BooleanScheduler(true_intervals=[[10, 60]], default=False),
        'skip_move': False,
        # 'w_eval': ExponentialScheduler(15, 0.2, 30, 0.8)
        'use_time_control': False,

    }
    p0_policy = SearchPolicy(AB_TT_Search_TC_v2(), config_p0)

    config_p1 = {
        'evaluate':evaluate_rel,
        'move_ordering':move_ordering,
        'depth': 30,
        'use_iterative_deepening': True,
        'deterministic': BooleanScheduler(true_intervals=[[10, 60]], default=False),
        'skip_move': False,
        'use_time_control': True,
        'budget_scheduler': Budget_Scheduler(num_turns=60, center=32, scale=5, alpha=1, p=0.3, w_2=1.7)
    }
    p1_policy = SearchPolicy(AB_TT_Search_TC_v2(), config_p1)
    env.render_mode = 'rgb_array'

    # print(SimulateEpisode(env=env, p0_policy=p0_policy, p1_policy=p1_policy, verbose=True))

    Evaluation_logs, Actions_logs, Policy_logs = SimulateMultipleEpisodes(env, p0_policy, p1_policy, n_episodes=8, verbose=False)
    save_path = os.path.join(BASE_SAVE_PATH, run_name)
    save_sim_logs(Evaluation_logs, Actions_logs, Policy_logs, save_path=save_path)

