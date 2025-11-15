from collections import defaultdict
import numpy as np
import pygame

import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from DotsAndBoxes import DnBEnv
from Policy import *
from Search import *

import tqdm.auto as tqdm

import os

import time
import json

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
    while not episode_over:
        cur_player =  observation['cur_player']
        policy = p0_policy if cur_player == 0 else p1_policy

        t0 = time.perf_counter()
        action, val = policy.get_action(observation, info, env)
        Policy_log.append(policy.get_log())
        t1 = time.perf_counter()

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
        scores.append(info['score'])

        Action_log.append(action)

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
    Action_log = {'Action_log': ", ".join(str(x) for a in Action_log for x in a)}
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
        Policy_logs.append(Policy_log)

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

def calc_basic_stats_from_eval_logs(Evaluation_logs):
    """
    Evaluation_logs: SimulateMultipleEpisodes에서 생성된 로그 리스트.
    
    반환: 한 줄짜리 pandas.DataFrame
      - n_episodes: 총 판수
      - n_p0_wins, n_p1_wins
      - p0_win_rate, p1_win_rate
      - p0_score_*: describe() 결과(prefix 붙임)
      - p1_score_*: describe() 결과(prefix 붙임)
      - p0_time_per_game, p1_time_per_game: 플레이어별 '판당' 평균 시간
      - p0_time_per_turn, p1_time_per_turn: 플레이어별 '턴당' 평균 시간
    """

    episode_rows = []

    for ep in Evaluation_logs:
        ep_id = ep.get("episode_id", None)
        log = ep.get("Evaluation_log")

        winner = log["winner"]
        scores_list = log["scores"]
        # 각 에피소드의 최종 점수
        final_p0, final_p1 = scores_list[-1]

        players = log["player"]
        times = log["time_spent"]

        # 플레이어별 총 시간 / 턴 수
        p0_total_time = sum(t for t, p in zip(times, players) if p == 0)
        p1_total_time = sum(t for t, p in zip(times, players) if p == 1)

        p0_n_turns = sum(1 for p in players if p == 0)
        p1_n_turns = sum(1 for p in players if p == 1)

        episode_rows.append({
            "episode_id": ep_id,
            "winner": winner,
            "p0_score": final_p0,
            "p1_score": final_p1,
            "p0_total_time": p0_total_time,
            "p1_total_time": p1_total_time,
            "p0_n_turns": p0_n_turns,
            "p1_n_turns": p1_n_turns,
        })

    df = pd.DataFrame(episode_rows)

    if len(df) == 0:
        raise ValueError("Evaluation_logs가 비어 있습니다.")

    # --- 기본 승수/승률 ---
    n_episodes = len(df)
    n_p0_wins = (df["winner"] == 0).sum()
    n_p1_wins = (df["winner"] == 1).sum()

    p0_win_rate = n_p0_wins / n_episodes
    p1_win_rate = n_p1_wins / n_episodes

    # --- 점수 describe ---
    p0_score_stats = df["p0_score"].describe().add_prefix("p0_score_")
    p1_score_stats = df["p1_score"].describe().add_prefix("p1_score_")

    # --- 시간 통계 ---
    # 판당 평균 시간: 각 에피소드의 total_time을 episode 축으로 평균
    p0_time_per_game = df["p0_total_time"].mean()
    p1_time_per_game = df["p1_total_time"].mean()

    # 턴당 평균 시간: 전체 에피소드의 total_time 합 / 전체 턴 수 합
    p0_total_time_all = df["p0_total_time"].sum()
    p1_total_time_all = df["p1_total_time"].sum()

    p0_total_turns_all = df["p0_n_turns"].sum()
    p1_total_turns_all = df["p1_n_turns"].sum()

    p0_time_per_turn = p0_total_time_all / p0_total_turns_all if p0_total_turns_all > 0 else np.nan
    p1_time_per_turn = p1_total_time_all / p1_total_turns_all if p1_total_turns_all > 0 else np.nan

    summary = {
        "n_episodes": n_episodes,
        "n_p0_wins": n_p0_wins,
        "n_p1_wins": n_p1_wins,
        "p0_win_rate": p0_win_rate,
        "p1_win_rate": p1_win_rate,
        "p0_time_per_game": p0_time_per_game,
        "p1_time_per_game": p1_time_per_game,
        "p0_time_per_turn": p0_time_per_turn,
        "p1_time_per_turn": p1_time_per_turn,
    }

    summary_df = pd.DataFrame([summary])

    # describe 결과 붙이기
    p0_score_df = p0_score_stats.to_frame().T.reset_index(drop=True)
    p1_score_df = p1_score_stats.to_frame().T.reset_index(drop=True)

    stats_df = pd.concat([summary_df, p0_score_df, p1_score_df], axis=1)

    return stats_df

def calc_basic_stats_from_policy_logs(Policy_logs):
    """
    Policy_Log을 기반으로 player별 key 통계를 계산하여
    player_id -> DataFrame 형태로 반환한다.

    각 DataFrame은 index = key, columns = [mean, std, min, max, count].
    """

    # --- 플레이어별 컨테이너 ---
    # per_player_values[player_id][key] = list of float values
    per_player_values: Dict[Any, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    # --- flatten & collect values ---
    for ep in Policy_logs:
        plog: Optional[Dict[str, Any]] = ep.get("Policy_log", {})
        players = plog.get("player", [])
        logs = plog.get("logs", [])

        L = min(len(players), len(logs)) if players else len(logs)

        for t in range(L):
            player_id = players[t] if players else None
            log_entry = logs[t]
            if log_entry == None:
                continue
            for key, val in log_entry.items():
                if val is None:
                    continue
                if isinstance(val, bool):
                    continue
                if isinstance(val, (int, float)):
                    per_player_values[player_id][key].append(float(val))

    # --- 통계 계산 + DataFrame 생성 ---
    player_stats: Dict[Any, pd.DataFrame] = {}

    stats_dict: Dict[str, Dict[str, float]] = {}

    for player_id, key_dict in per_player_values.items():

        for key, vals in key_dict.items():
            if len(vals) == 0:
                continue
            arr = np.asarray(vals, dtype=float)
            prefix = 'p0_' if player_id == 0 else 'p1_'
            key = prefix + key
            stats_dict[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "count": int(arr.size),
            }



    df_stats = pd.DataFrame.from_dict(stats_dict, orient="index")
    # 컬럼 순서 고정
    if len(df_stats) > 0:
        df_stats = df_stats[["mean", "std", "min", "max", "count"]]

    return df_stats

def save_sim_logs(Evaluation_logs, Actions_logs, Policy_logs, run_name: str):
    """
    SimulateMultipleEpisodes의 결과를 지정된 경로에 저장합니다.
    
    이거보단 나은 저장방식이 있을 것 같은데 Todo에 부치겠습니다.
    """

    save_path = os.path.join(BASE_SAVE_PATH, run_name)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'Action_logs.csv'), 'w') as f:
        json.dump(Actions_logs, f)

    with open(os.path.join(save_path, 'Evaluation_logs.csv'), 'w') as f:
        json.dump(Evaluation_logs, f)

    with open(os.path.join(save_path, 'Policy_logs.csv'), 'w') as f:
        json.dump(Policy_logs, f)

    basic_eval_stat_df = calc_basic_stats_from_eval_logs(Evaluation_logs=Evaluation_logs)
    basic_eval_stat_df.to_csv(os.path.join(save_path, 'Evaluation_stats.csv'))
    
    basic_policy_stat_df = calc_basic_stats_from_policy_logs(Policy_logs=Policy_logs)
    basic_policy_stat_df.to_csv(os.path.join(save_path, 'Policy_stats.csv'))
    
    print(f"Simulation results saved to {save_path}")

if __name__ == "__main__":

    run_name = 'skip_vs_no_skip'
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
        'move_ordering':None,
        'depth': ExponentialSchedulerInt(15, 2, 35, 5),
        'use_iterative_deepening': True,
        'deterministic': BooleanScheduler(true_intervals=[[10, 60]], default=False),
        'skip_move': True
    }
    p0_policy = SearchPolicy(AB_TT_Search(), config_p0)
    config_p1 = {
        'evaluate':evaluate_rel,
        'move_ordering':None,
        'depth': ExponentialSchedulerInt(15, 2, 35, 5),
        'use_iterative_deepening': True,
        'deterministic': BooleanScheduler(true_intervals=[[10, 60]], default=False),
        'skip_move': False
    }
    p1_policy = SearchPolicy(AB_TT_Search(), config_p1)

    # SimulateEpisode(env=env, p0_policy=p0_policy, p1_policy=p1_policy, verbose=True)

    env.render_mode = 'rgb_array'
    Evaluation_logs, Actions_logs, Policy_logs = SimulateMultipleEpisodes(env, p0_policy, p1_policy, n_episodes=20, verbose=False)
    save_sim_logs(Evaluation_logs, Actions_logs, Policy_logs, run_name=run_name)

