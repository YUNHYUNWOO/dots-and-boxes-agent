# main.py에 대한 예시 파일 안내입니다!

# import torch
# import os
import random
from Policy import EndgamePolicy
import numpy as np

model = None  # 전역 변수로 모델을 유지해주세요! 변수 이름도 model로 유지하셔야 합니다!

# 사용할 모델의 구조를 지정하는 클래스나 사용할 보조 함수를 여기에 작성하세요.
# 예시1: 모델 구조 정의 클래스
# class MyModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # 모델의 레이어를 정의하세요.
#     def forward(self, x):
#         # 모델의 순전파 과정을 정의하세요.
#         return x

# 예시2: run에서 사용할 보조 함수
# def helper_function(args):
#     # 보조 함수의 내용을 작성하세요.

# 반드시 init(),run()함수를 구현해줘야 합니다. 없으면 에러가 발생합니다.
def init():
    # << 체점 시 양쪽 에이전트에 대해서 처음 한 번 실행되는 함수입니다. >>
    # 딥러닝을 통해 게임 에이전트 모델을 training하신 경우에는 모델을 델러오고, 평가 모드로 전환하는 부분을 이곳에 넣으셔야 합니다.
    # 딥러닝을 사용하지 않으셨더라도, Model-based AI로 에이전트를 만드신 분들도 이곳에서 모델/데이터 로딩을 하시면 됩니다.
    global model
    
    # 예시1: 학습된 모델 로드
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(current_dir, "weights.pt") # 학습된 모델 파일 이름을 작성하세요.
    # model = torch.load(model_path, map_location="cpu") 
    # 훈련한 모델을 불러오는 경우 *반드시* 위의 방법으로 상대 경로를 지정하여 불러오시기 바랍니다. 양식을 따르지 않을 경우 채점 서버에서 오류가 발생할 수 있습니다.
    # model.eval() # model을 training이 아닌 evalutation 모드로 전환
    
    # 예시2: 학습된 모델을 사용하지 않는 경우
    model = None 
    
    # 위의 코드는 모델을 사용하지 않는다는 의미입니다. 모델이 필요없는 Rule-based AI를 구현하신 분들은 이렇게 작성하시면 됩니다.

def run(board_lines, xsize, ysize):
    # << 에이전트의 차례가 될 때마다 실행되는 함수입니다. >>
    # 함수의 입력은 위와 같이 현재 board의 현재 상태 (놓인 수들)과 보드의 크기가 제공됩니다.
    # board_lines는 3차원 리스트의 형태로, board_lines[x][y][z]은 해당 자리(x, y, z는 아래 설명 참고)에 수가 놓였는지, 놓이지 않았는지에 대한 값으로 0 또는 1을 가집니다.
    # 이러한 입력 값을 바탕으로, 다음과 같이 놓을 수를 반환해주시면 됩니다.

    def convert_action_to_submit_format(action: tuple[int, int, int]) -> tuple[int, int, int]:
        return (action[2], action[1], action[0])

    def convert_board_to_observation(board_lines):
        h_edges = [[0 for __ in range(xsize)] for _ in range(ysize + 1)]
        v_edges = [[0 for __ in range(ysize + 1)] for _ in range(xsize)]

        for z in range(2):
            for r in range(ysize + 1):
                for c in range(xsize + 1):
                    if z == 0 and c != xsize:
                        h_edges[r][c] = board_lines[c][r][z]
                    elif z == 1 and r != ysize:
                        v_edges[r][c] = board_lines[c][r][z]    

        observation = {
            'h_edges': h_edges,
            'v_edges': v_edges,
            'box_owner': None
        }

        return observation

    def convert_board_lines_to_action_mask(board_lines) -> np.ndarray[bool]:
        n_boxes = len(board_lines) - 1
        
        def get_init_action_mask(n_box) -> np.ndarray:
            # shape: (n_box+1, n_box+1)
            r = np.arange(n_box + 1)[:, None]        # (R,1)
            c = np.arange(n_box + 1)[None, :]        # (1,C)
            mask = np.zeros((2, n_box + 1, n_box + 1), dtype=bool)
            # Horizontal mask: True only when c == n_box
            
            mask[0] = (c == n_box)
            # Vertical mask: True only when r == n_box
            mask[1] = (r == n_box)
            # Stack so that mask[0] = H, mask[1] = V
            
            return mask

        action_mask = get_init_action_mask(n_boxes)
        for r in range(ysize + 1):
            for c in range(xsize + 1):
                for z in range(2):
                    if board_lines[c][r][z]:
                        action_mask[z][r][c] = True

        return action_mask

    policy = EndgamePolicy()

    observation = convert_board_to_observation(board_lines)
    # print("Observation:")
    # print(observation)

    info = {
        'action_mask': convert_board_lines_to_action_mask(board_lines)
    }
    # print("action_mask:")
    # print(info['action_mask'])

    action = policy.get_action(observation=observation, info=info, env=None)

    return convert_action_to_submit_format(action)

if __name__ == "__main__":
    for _ in range(100000):
        board_lines = [[[0 for _ in range(2)] for __ in range(6)]  for ___ in range(6)]

        for c in range(6):
            for r in range(6):
                for z in range(2):
                    board_lines[c][r][z] = random.randint(0, 1)
                    
        # for z in range(2):
        #     for r in range(6):
        #         for c in range(6):
        #             if z == 0 and c != 5:
        #                 print(board_lines[c][r][z], end=' ')
        #             elif z == 1 and r != 5:
        #                 print(board_lines[c][r][z], end=' ')
        #         print()
                
        # print("Board Lines:")
        # print(board_lines)

        action = run(board_lines, 5, 5)
        if board_lines[action[0]][action[1]][action[2]]:
            print("Selected Action:")
            print(action)
            break   
