from main import run
import random

if __name__ == "__main__":

    board_lines = [[[0 for _ in range(2)] for __ in range(6)]  for ___ in range(6)]

    for c in range(6):
        for r in range(6):
            for z in range(2):
                board_lines[c][r][z] = random.randint(0, 1)
    
    print("Board Lines:")
    print(board_lines)

    print(run(board_lines, 5, 5))