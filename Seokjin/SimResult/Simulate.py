
import numpy as np
import pygame

import pandas as pd

import gymnasium as gym
from gymnasium import spaces

from DotsAndBoxes import DnBEnv
from Policy import *

import os

BASE_SAVE_PATH = './SimResult/'

# 정책에 의한 전체 에피소드를 시뮬레이션 하는 함수
# 만약 pygame window가 작동하지 않으면 env 생성시 render_mode를 'human'으로 설정할 것
def SimulateEpisode(env, p0_policy: Policy, p1_policy: Policy, verbose=False):
    # verbose는 디버깅 출력 여부
    observation, info = env.reset()
    action_mask = info['action_mask']

    if verbose:
        print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0

    while not episode_over:
        if observation['cur_player'] == 0:
            policy = p0_policy
        else:
            policy = p1_policy
            
            observation['box_owner'] = (-1 * np.array(observation['box_owner'])).tolist()  # p1 입장에서 box_owner 정보 반전

        action = policy.get_action(observation, info, env)
        
        if verbose:
            print('action:', action)
            print(info['action_mask'])
            print('Number of Claimed Edges:', np.sum(action_mask == False))

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    if verbose:
        print(f"Episode finished! Winner: {info['winner']} Total reward: {total_reward}")
        print(f'Action spasce: {env.action_space}')
        print(f'Observation spasce: {env.observation_space}')

    env.close()

    return observation, info, total_reward

def SimulateMultipleEpisodes(env, p0_policy: Policy, p1_policy: Policy, n_episodes: int, verbose=False):
    results = {
        'observation': [],
        'info': [],
        'total_reward': []
    }

    for episode in range(n_episodes):
        if verbose:
            print(f"=== Episode {episode + 1} ===")
        observation, info, total_reward = SimulateEpisode(env, p0_policy, p1_policy, verbose)
        results['observation'].append(observation)
        results['info'].append(info)
        results['total_reward'].append(total_reward)
    
    return results

def calc_basic_stats(results):

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
    save_path = os.path.join(BASE_SAVE_PATH, run_name)
    os.makedirs(save_path, exist_ok=True)

    infos_df = pd.DataFrame(results['info'])
    infos_df.to_csv(os.path.join(save_path, 'infos.csv'), index=False)

    observations_df = pd.DataFrame(results['observation'])
    observations_df.to_csv(os.path.join(save_path, 'observations.csv'), index=False)

    rewards_df = pd.DataFrame({'total_reward': results['total_reward']})
    rewards_df.to_csv(os.path.join(save_path, 'rewards.csv'), index=False)

    stats_df = calc_basic_stats(results)
    stats_df.to_csv(os.path.join(save_path, 'stats.csv'), index=False)

    print(f"Simulation results saved to {save_path}")

# if __name__ == "__main__":

#     run_name = 'Random_vs_FixedOrder'
#     n_box = 5
#     env = DnBEnv(render_mode='rgb_array', n_box=n_box)
    
#     p0_policy = RandomPolicy()
#     p1_policy = EndgamePolicy()
#     # policy = RandomPolicy()
#     results = SimulateMultipleEpisodes(env, p0_policy, p1_policy, n_episodes=1000, verbose=False)
#     save_simulation_results(results, run_name=run_name)

if __name__ == "__main__":

    run_name = 'Random_vs_FixedOrder'
    n_box = 5
    env = DnBEnv(render_mode='human', n_box=n_box)

    results = SimulateEpisode(env=env, p0_policy=FixedOrderPolicy(5), p1_policy=EndgamePolicy(), verbose=True)

    env.render_mode = 'rgb_array'
    p0_policy = FixedOrderPolicy(5)
    p1_policy = EndgamePolicy()

    # policy = RandomPolicy()
    results = SimulateMultipleEpisodes(env, p0_policy, p1_policy, n_episodes=1000, verbose=False)
    save_simulation_results(results, run_name=run_name)
