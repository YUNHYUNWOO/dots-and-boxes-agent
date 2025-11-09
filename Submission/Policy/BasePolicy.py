import numpy as np

class BasePolicy():
    def __init__(self):
        ## 필요한거 있으면 추가
        pass

    def get_action(self, observation, info, env):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음
        raise NotImplementedError


## 예시 정책
class RandomPolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def get_action(self, observation, info, env):
        action_mask = info['action_mask']
        action = env.action_space.sample()
        # info['action_mask']가 True인 액션은 이미 선택된 액션이므로 다시 샘플링
        while action_mask[action[0], action[1], action[2]]:
            action = env.action_space.sample()

        return action


class FixedOrderPolicy(BasePolicy):
    def __init__(self, n_box):
        super().__init__()
        self.n_box = n_box
        self.action_order = []
        for ori in range(2):
            for r in range(n_box + 1):
                for c in range(n_box + 1):
                    if (ori == 0 and c == n_box) or (ori == 1 and r == n_box):
                        continue
                    self.action_order.append((ori, r, c))
        self.current_index = 0


    def get_action(self, observation, info, env):
        action = self.action_order[self.current_index]

        self.current_index = (self.current_index + 1) % len(self.action_order)
        return action


# 정책에 의한 전체 에피소드를 시뮬레이션 하는 함수
# 만약 pygame window가 작동하지 않으면 env 생성시 render_mode를 'human'으로 설정할 것
def SimulateEpisode(env, policy: BasePolicy, verbose=False):
    # verbose는 디버깅 출력 여부
    observation, info = env.reset()
    action_mask = info['action_mask']

    if verbose:
        print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0

    while not episode_over:
        action = policy.get_action(observation, info, env)
        
        if verbose:
            print('action:', action)
            print(info['action_mask'])
            print('Number of Claimed Edges:', np.sum(action_mask == False))

        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated

    if verbose:
        print(f"Episode finished! Total reward: {total_reward}")
        print(f'Action spasce: {env.action_space}')
        print(f'Observation spasce: {env.observation_space}')

    env.close()
