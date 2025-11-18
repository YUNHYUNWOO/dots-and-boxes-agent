from typing import List, Tuple, Dict, Optional
import random
import time
import pandas

N_BOX = 5
N = N_BOX + 1
H_COUNT = N * (N - 1)   # 30
V_COUNT = (N - 1) * N   # 30
TOTAL_BOXES = N_BOX * N_BOX  # 25
H, V = 0, 1

def h_index(c: int, r: int) -> int:
    return r * (N - 1) + c

def v_index(c: int, r: int) -> int:
    return r * N + c

def check_bounds(c: int, r: int, d: int) -> None:
    if not (0 <= r < N and 0 <= c < N and d in (H, V)):
        raise ValueError("Action out of bounds: r,c in [0,5], d in {0,1}")
    if d == H and c >= N - 1:
        raise ValueError("Invalid H edge: c must be <= 4")
    if d == V and r >= N - 1:
        raise ValueError("Invalid V edge: r must be <= 4")

def boxes_adjacent_to_edge(c: int, r: int, d: int) -> List[Tuple[int, int]]:
    if d == H:
        if 0 <= r - 1 < N_BOX: return [(c, r - 1), (c, r)]
        elif r < N_BOX: return [(c, r)]
    else:
        if 0 <= r < N_BOX and 0 <= c - 1 < N_BOX: return [(c - 1, r), (c, r)]
        if 0 <= r < N_BOX and c < N_BOX: return [(c, r)]

def is_box_complete(h_bits: int, v_bits: int, bc: int, br: int) -> bool:

    #h1 = (h_bits >> h_index(bc, br)) & 1
    #h2 = (h_bits >> h_index(bc, br + 1)) & 1
    #v1 = (v_bits >> v_index(bc, br)) & 1
    #v2 = (v_bits >> v_index(bc + 1, br)) & 1

    # h1 = br * (N - 1) + bc
    # h2 = (br + 1) * (N - 1) + bc
    # v1 = br * N + bc
    # v2 = br * N + bc + 1

    return (h_bits >> (br * (N - 1) + bc) & 1) & (h_bits >> ((br + 1) * (N - 1) + bc) & 1) & (v_bits >> (br * N + bc) & 1) & (v_bits >> (br * N + bc + 1) & 1)

def count_completed_boxes(h_bits: int, v_bits: int) -> int:
    cnt = 0
    for br in range(N_BOX):
        for bc in range(N_BOX):
            if (h_bits >> (br * (N - 1) + bc) & 1) & (h_bits >> ((br + 1) * (N - 1) + bc) & 1) & (v_bits >> (br * N + bc) & 1) & (v_bits >> (br * N + bc + 1) & 1):
                cnt += 1
    return cnt

def edge_is_claimed(h_edge:int, v_edge:int, c: int, r: int, d: int) -> bool:
    if d == H: return ((h_edge >> (r * (N - 1) + c)) & 1)
    else:      return ((v_edge >> (r * N + c)) & 1)

def edges_adjacent_to_box(c, r):
    return [
        [c, r, 0],
        [c + 1, r, 1],
        [c, r + 1, 0],
        [c, r, 1]
    ]
def encode_Edges(edges):
    """
        edges를 받아 int 2개로 압축한다.
        BitMask 사용
    """
    h, v = 0, 0
    for r in range(N):
        for c in range(N):
            #print(j, edges[j][i])
            if c != N-1 and edges[c][r][H]: 
                h |= 1 << h_index(c, r)
            if r != N-1 and edges[c][r][V]:
                v |= 1 << v_index(c, r)
                
    return [h,v]


def decode_Edges(encoded_edges):
    """
    encoding 방식이 n_box에 유연하게 대처하지 못하게 되어있음
    n_box가 파라미터로 되어있지만 n_box를 빼는 순간 오류가 날 것
    """
    h, v = encoded_edges

    edges = [[[0 for _ in range(2)] for _ in range(N)] for _ in range(N)]


    for r in range(N):
        for c in range(N):
            if c != N-1: 
                edges[c][r][H] = ((h >> h_index(c, r)) & 1) == 1

            if r != N-1: 
                edges[c][r][V] = ((v >> v_index(c, r)) & 1) == 1

    return edges

def get_legal_actions(encoded_edges, n_box=5):
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

def _test_get_available_actions_encoded():
    for i in range(10000):
        h = random.randint(0, 1 << 30)
        v = random.randint(0, 1 << 30)

        encoded_edges = [h, v]
        actions = get_legal_actions(encoded_edges=encoded_edges)

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
    print(encode_Edges(edges), h_index(1, 3))

    # test Edge Encode, Decode
    _test_encode_decode()
    _test_get_available_actions_encoded()
        
