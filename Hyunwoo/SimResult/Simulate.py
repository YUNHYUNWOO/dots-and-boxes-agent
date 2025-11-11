
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

BASE_SAVE_PATH = './SimResult/'

def convert_action_to_submit_format(action: tuple[int, int, int]) -> tuple[int, int, int]:
    return (action[1], action[2], action[0])

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
    action_mask = info['action_mask']
    record = []

    if verbose:
        print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0

    while not episode_over:
        if observation['cur_player'] == 0:
            policy = p0_policy
        else:
            policy = p1_policy

        action = policy.get_action(observation, info, env)
        record.append(convert_action_to_submit_format(action))

        if verbose:
            print('action:', action)
            print('action_mask:')
            for i in range(2):
                ori = 'H' if i == 0 else 'V'
                print(f'--{ori}--')
                for r in range(info['action_mask'].shape[0]):
                    print(info['action_mask'][:,r,i])
                print('-----')

            print('Number of Claimed Edges:', np.sum(action_mask == False))

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    if verbose:
        print(f"Episode finished! Winner: {info['winner']} Total reward: {total_reward}")
        print(f'Action spasce: {env.action_space}')
        print(f'Observation spasce: {env.observation_space}')

    env.close()

    return record, info, total_reward


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
        'total_reward': []
    }

    first_plaer = p0_policy
    second_player = p1_policy

    for episode in tqdm.tqdm(range(n_episodes)):
        if verbose:
            print(f"=== Episode {episode + 1} ===")
        record, info, total_reward = SimulateEpisode(env, first_plaer, second_player, verbose)

        results['record'].append(record)
        results['info'].append(info)
        results['total_reward'].append(total_reward)
        results['info'][-1]['first_player'] = 0

        if first_plaer == p1_policy:
            results['total_reward'][-1] *= -1  # 후공 입장에서의 보상으로 변환
            results['info'][-1]['winner'] = 1 - results['info'][-1]['winner']  # 승자 정보도 변환
            results['info'][-1]['first_player'] = 1
            results['info'][-1]['score'] = results['info'][-1]['score'][::-1]

        first_plaer, second_player = second_player, first_plaer  # 다음 에피소드에서 선후공 교체
        

    return results

def calc_basic_stats(results):
    """
    SimulateMultipleEpisodes의 결과로부터 기본 통계치를 계산합니다.
    Returns:
        stats: pandas DataFrame 형태로 반환됩니다.
            n_episodes: 총 에피소드 수
            n_p0_wins: 첫 번째 정책이 이긴 에피소드 수
            n_p1_wins: 두 번째 정책이 이긴 에피소드 수
            p0_win_rate: 첫 번째 정책의 승률
            p0_score_*: 첫 번째 정책의 점수에 대한 기본 통계량(mean, std, 25%, 50%, 75% 등)
            p1_score_*: 두 번째 정책의 점수에 대한 기본 통계량(mean, std, 25%, 50%, 75% 등)
    """
    infos_df = pd.DataFrame(results['info'])

    n_episodes = len(infos_df)
    n_p0_wins = (infos_df['winner'] == 0).sum()
    n_p1_wins = (infos_df['winner'] == 1).sum()

    stats = pd.DataFrame({
        'n_episodes': [n_episodes,],
        'n_p0_wins': [n_p0_wins,],
        'n_p1_wins': [n_p1_wins,],
        'p0_win_rate': [n_p0_wins / n_episodes]
    })
    
    p0_basic_stats = infos_df['score'].apply(lambda x: x[0]).describe().add_prefix('p0_score_').to_frame().T.reset_index(drop=True)
    p1_basic_stats = infos_df['score'].apply(lambda x: x[1]).describe().add_prefix('p1_score_').to_frame().T.reset_index(drop=True)

    stats = pd.concat([stats, p0_basic_stats, p1_basic_stats], axis=1)

    return stats

def save_simulation_results(results, run_name: str):
    """
    SimulateMultipleEpisodes의 결과를 지정된 경로에 저장합니다.
    
    이거보단 나은 저장방식이 있을 것 같은데 Todo에 부치겠습니다.
    """
    save_path = os.path.join(BASE_SAVE_PATH, run_name)
    os.makedirs(save_path, exist_ok=True)

    infos_df = pd.DataFrame(results['info'])
    infos_df.to_csv(os.path.join(save_path, 'infos.csv'), index=False)

    observations_df = pd.DataFrame(results['record'])
    observations_df.to_csv(os.path.join(save_path, 'record.csv'), index=False)

    rewards_df = pd.DataFrame({'total_reward': results['total_reward']})
    rewards_df.to_csv(os.path.join(save_path, 'rewards.csv'), index=False)

    stats_df = calc_basic_stats(results)
    stats_df.to_csv(os.path.join(save_path, 'stats.csv'), index=False)

    print(f"Simulation results saved to {save_path}")

if __name__ == "__main__":

    run_name = 'AlphaBeta_v1_d3_vs_AlphaBeta_v1_d4'
    n_box = 5
    env = DnBEnv(render_mode='human', n_box=n_box)
    
    alphabeta_search_d3 = AlphaBetaSearch(evaluate=evaluate, move_ordering=None, depth=2)
    p0_policy = Search_Policy(SearchEngine=alphabeta_search_d3)
    # p0_policy = FixedOrderPolicy(5)
    alphabeta_search_d4 = AlphaBetaSearch(evaluate=evaluate, move_ordering=None, depth=3)
    p1_policy = Search_Policy(SearchEngine=alphabeta_search_d4)

    results = SimulateEpisode(env=env, p0_policy=p0_policy, p1_policy=p1_policy, verbose=True)

    env.render_mode = 'rgb_array'

    results = SimulateMultipleEpisodes(env, p0_policy, p1_policy, n_episodes=10, verbose=False)
    save_simulation_results(results, run_name=run_name)
