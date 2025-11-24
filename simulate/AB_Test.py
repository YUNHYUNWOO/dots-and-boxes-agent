import os
import time

from dotsandboxes import DnBEnv
from policy import *
from search import *
from util import *


def Test_AB_Model(env, A_policy: BasePolicy, B_policy: BasePolicy, verbose=False):
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

    episode_over = False
    A_time_manager = TimeManager()
    B_time_manager = TimeManager()

    while not episode_over:
        t0 = time.perf_counter()
        A_time_manager.start_move()
        A_action, A_val = A_policy.get_action(observation, info, env, A_time_manager)
        A_time_manager.end_move()        
        t1 = time.perf_counter()  

        t0 = time.perf_counter()
        B_time_manager.start_move()
        B_action, B_val = B_policy.get_action(observation, info, env, A_time_manager)
        B_time_manager.end_move()        
        t1 = time.perf_counter()

        print(f'A_action: {A_action}, B_action: {B_action}')
        print(f'A_val: {A_val}, B_val: {B_val}')
        print(f'A_log: {A_policy.get_log()}, B_log: {B_policy.get_log()}')

        observation, _, terminated, truncated, info = env.step(A_action)
        episode_over = terminated or truncated
    env.close()


