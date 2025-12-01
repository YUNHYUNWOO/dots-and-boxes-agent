import os
import random

import numpy as np
import tqdm.auto as tqdm
import pygame
import json

from dotsandboxes import DnBEnv
from config import P0, P1
from config.load_config import load_config, make_obejct_from_config
from policy import BasePolicy, TimeManager
from util import *
from .logger import EpisodeLogger, MultiEpisodeLogger

BASE_SAVE_PATH = './Simulate/SimResult'


# 정책에 의한 전체 에피소드를 시뮬레이션 하는 함수
# 만약 pygame window가 작동하지 않으면 env 생성시 render_mode를 'human'으로 설정할 것
def simulate_episode(env: DnBEnv, p0_policy: BasePolicy, p1_policy: BasePolicy, verbose=False):
    """
    """
        
    # verbose는 디버깅 출력 여부
    observation, info = env.reset()

    if verbose:
        print(f"Starting observation: {observation}")

    ep_logger = EpisodeLogger()
    episode_over = False
    p0_time_manager = TimeManager()
    p1_time_manager = TimeManager()

    while not episode_over:
        cur_player =  observation['cur_player']

        time_manager = p0_time_manager if cur_player == 0 else p1_time_manager
        policy = p0_policy if cur_player == 0 else p1_policy

        if verbose:
            print('cur_player: ', cur_player)

        time_manager.start_move()
        action = policy.get_action(observation, time_manager)
        time_spent = time_manager.end_move()        

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

        observation, _, terminated, _, info = env.step(action)
        episode_over = terminated
        
        ep_logger.log(player=cur_player, 
                      time_spent=time_spent, 
                      scores=list(info['score']), 
                      action=(cur_player, *action),
                      policy_log=policy.get_log())
        
    if verbose:
        print(f"Episode finished! Winner: {info['winner']}")
        print(f'Action spasce: {env.action_space}')
        print(f'Observation spasce: {env.observation_space}')

    env.close()
   
    return (
        ep_logger.get_default_log(info['winner']), 
        ep_logger.get_action_log(), 
        ep_logger.get_policy_log()
    )
  


def simulate_multiple_episodes(env: DnBEnv, p0_policy: BasePolicy, p1_policy: BasePolicy, n_episodes: int, log: bool, save_path: str, verbose: bool=False):
    """
    Run multiple self-play episodes between two policies in the Dots and Boxes environment.

    For each episode this function:
    - simulates a full game between `first_player` and `second_player` using `simulate_episode`
    - logs default, action, and policy logs with `MultiEpisodeLogger`
    - alternates which policy takes the first move in the next episode

    Args:
        env: DnBEnv game environment.
        p0_policy: Policy used as Player 0.
        p1_policy: Policy used as Player 1.
        n_episodes: Number of episodes to simulate.
        log: Flag indicating whether logging should be enabled (currently not used).
        save_path: Directory where logs and aggregated statistics will be saved.
        verbose: If True, print per-episode progress messages.

    Returns:
        None
    """

    first_player = p0_policy
    second_player = p1_policy


    logger = MultiEpisodeLogger(save_path) if log else None

    for episode in tqdm.tqdm(range(n_episodes)):
        if verbose:
            print(f"=== Episode {episode + 1} ===")
        default_log, action_log, policy_log  = simulate_episode(env, first_player, second_player, verbose)

        if log:
            logger.log(default_log, action_log, policy_log, episode_id=episode, first_player_is_p0=(first_player is p0_policy))

        first_player, second_player = second_player, first_player  # 다음 에피소드에서 선후공 교체
    
    if log:
        logger.log_stats()
    return


def run_config(config_path: str, render_mode: str):
    run_name, n_episodes, p0_policy, p1_policy, config_json = load_config(config_path)
    save_path = os.path.join(BASE_SAVE_PATH, run_name)

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_json, f, indent=4)

    env = DnBEnv(render_mode=render_mode)
    simulate_multiple_episodes(env=env, 
                               p0_policy=p0_policy, 
                               p1_policy=p1_policy,
                               n_episodes=n_episodes, 
                               log=True, 
                               save_path=save_path, 
                               verbose=False)

def play_against_policy(config_path: str, user_first: bool):
    waiting = True

    env = DnBEnv(render_mode='human')


    with open(config_path, mode='r') as f:
        config = json.load(f)
    policy = make_obejct_from_config(config)
    time_manager = TimeManager()
    observation, info = env.reset()
    dnb = env.DnB
    episode_over = False
    while not episode_over:
        if user_first and observation['cur_player'] == P0:
            waiting = True
            while waiting:
                mouse_pos = pygame.mouse.get_pos()
                dnb.hover_edge = dnb.find_hover_edge(mouse_pos)
                env._render_frame()
                
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        e = dnb.find_hover_edge(event.pos)
                        if e is not None:
                            d, r, c = e
                            action = (c, r, 0 if d == 'H' else 1)
                            observation, _, terminated, _, info = env.step(action)
                            episode_over = terminated
                            waiting = False
                        
        else:
            time_manager.start_move()
            action = policy.get_action(observation, time_manager)
            time_manager.end_move() 
            observation, _, terminated, _, info = env.step(action)
            episode_over = terminated  