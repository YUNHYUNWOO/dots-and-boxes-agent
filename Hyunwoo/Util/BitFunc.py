from typing import List, Tuple, Dict, Optional
import random
import time
import pandas
from DotsAndBoxes import DotsAndBoxesEngine, _h_index, _v_index


def encode_Edges(edges):
    """
        edges를 받아 int 2개로 압축한다.
        BitMask 사용
    """
    n = len(edges)
    h, v = 0, 0
    for i in range(n):
        for j in range(n):
            #print(j, edges[j][i])
            if j != n-1 and edges[j][i][0]: 
                h |= 1 << (j + i * (n - 1))
            if i != n-1 and edges[j][i][1]:
                v |= 1 << (j + i * n)
                
    return [h,v]


def decode_Edges(encoded_edges, n_box = 5):
    """
    encoding 방식이 n_box에 유연하게 대처하지 못하게 되어있음
    n_box가 파라미터로 되어있지만 n_box를 빼는 순간 오류가 날 것
    """
    h, v = encoded_edges
    n = n_box + 1

    edges = [[[0 for _ in range(2)] for _ in range(n)] for _ in range(n)]
    

    for i in range(n):
        for j in range(n):
            if j != n-1: 
                edges[j][i][0] = 1 if (h & 1 << (j + i * (n-1))) else 0

            if i != n-1: 
                edges[j][i][1] = 1 if (v & 1 << (j + i * n)) else 0 

    return edges
def get_legal_actions_encoded(encoded_edges, n_box=5):
    """
    encoded_edges = [h, v]
      - H(r,c): r in [0..n-1], c in [0..n-2], idx = r*(n-1) + c
      - V(r,c): r in [0..n-2], c in [0..n-1], idx = r*n + c
    반환: [[r, c, d], ...]  (d: 0=H, 1=V)
    """
    n = n_box + 1
    h, v = encoded_edges

    # 유효 비트 길이로 마스킹 (초과 비트가 켜져 있어도 안전)
    H_COUNT = n * (n - 1)   # 수평 엣지 개수
    V_COUNT = (n - 1) * n   # 수직 엣지 개수
    h &= (1 << H_COUNT) - 1
    v &= (1 << V_COUNT) - 1

    actions = []

    # 수평 엣지들: r ∈ [0..n-1], c ∈ [0..n-2]
    # 각 행마다 (n-1)비트를 묶어서 읽으면 인덱스 계산 실수를 줄일 수 있음
    for r in range(n):
        row_bits = (h >> (r * (n - 1))) & ((1 << (n - 1)) - 1)
        for c in range(n - 1):
            if ((row_bits >> c) & 1) == 0:     # 아직 미설치
                actions.append([c, r, 0])      # d=0 (H)

    # 수직 엣지들: r ∈ [0..n-2], c ∈ [0..n-1]
    for r in range(n - 1):
        row_bits = (v >> (r * n)) & ((1 << n) - 1)
        for c in range(n):
            if ((row_bits >> c) & 1) == 0:     # 아직 미설치
                actions.append([c, r, 1])      # d=1 (V)

    return actions

def apply_acton(encoded_edges, move, n_box=5):
    c, r, z = move
    n = n_box + 1

    h, v = encoded_edges

    assert (~h & (1 << c + r * (n - 1))), "The action has already been taken. Choose another action."

    h_ = h | (1 << c + r * (n - 1))
    v_ = v | (1 << c + r * n)

    return [h, v]

def _test_get_available_actions_encoded():
    for i in range(10000):
        h = random.randint(0, 1 << 30)
        v = random.randint(0, 1 << 30)

        encoded_edges = [h, v]
        actions = get_legal_actions_encoded(encoded_edges=encoded_edges)

        actions_edges = [[[0 for _ in range(2)] for _ in range(6)] for _ in range(6)]

        for action in actions:
            c, r, z = action
            actions_edges[c][r][z] = 1

        for z in range(2):
            for r in range(6):
                for c in range(6):
                    print(actions_edges[c][r][z], end = " ")
                print()
            print("=====")

        decoded_edges = decode_Edges(encoded_edges)

        for z in range(2):
            for r in range(6):
                for c in range(6):
                    print(decoded_edges[c][r][z], end = " ")
                print()
            print("=====")
            

        h_comp, v_comp = encode_Edges(actions_edges)
        
        assert ((h_comp & h) == 0) and ((v_comp & v) == 0)



def _test_encode_decode():
    t_arr = []

    for i in range(100):
        start = time.time()

        h = random.randint(0, 1 << 30)
        v = random.randint(0, 1 << 30)

        edges = decode_Edges([h,v])
        h_, v_ = encode_Edges(edges)

        assert (h_ == h and v_ == v)
        print(h, h_, v, v_)
        end = time.time()
        t_arr.append((end - start))

    t_sr = pandas.Series(t_arr)
    print(t_sr.head())
    print(t_sr.mean(), t_sr.std())

if __name__ == "__main__":
    edges = [[[0 for _ in range(2)] for _ in range(6)] for _ in range(6)]
    edges[3][1][0] = 1
    print(encode_Edges(edges), _h_index(1, 3))

    # test Edge Encode, Decode
    _test_encode_decode()
    _test_get_available_actions_encoded()
        
