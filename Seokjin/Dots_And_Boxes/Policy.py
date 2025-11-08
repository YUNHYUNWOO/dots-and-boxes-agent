from enum import Enum

import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

import importlib

from DnB import DotsAndBoxes, get_render_desc, draw_board
from DnB_Env import DnBEnv, get_action_sample_mask

n_box = 5
env = DnBEnv(render_mode='human', n_box=n_box)
mask = get_action_sample_mask(n_box)

observation, info = env.reset()

print(f"Starting observation: {observation}")

def update_action_mask(observation):    
    for r in range(len(observation['h_edges'])):
        for c, e in enumerate(observation['h_edges'][r]):
            mask[0, r, c] |= e
    for r in range(len(observation['v_edges'])):
        for c, e in enumerate(observation['v_edges'][r]):
            mask[1, r, c] |= e
    
episode_over = False
total_reward = 0

action_order = []

for ori in range(2):
    for r in range(n_box + 1):
        for c in range(n_box + 1):
            if not mask[ori, r, c]:
                action_order.append((ori, r, c))


i = 0
while not episode_over:

    action = action_order[i]
    i += 1

    while mask[*action]:
        action = env.action_space.sample()
        
    print('action:', action)
    print('Number of Claimed Edges:', np.sum(mask == False))

    observation, reward, terminated, truncated, info = env.step(action)
    
    update_action_mask(observation)
            
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
print(f'Action spasce: {env.action_space}')
print(f'Observation spasce: {env.observation_space}')

env.close()