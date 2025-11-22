import random
import time
import pandas

from config import *

from .dnb_util import (
    get_boxes_adjacent_to_edge,
    get_edges_adjacent_to_box,
)

def h_index(c: int, r: int) -> int:
    return r * (N - 1) + c

def v_index(c: int, r: int) -> int:
    return r * N + c

def bit_is_box_complete(board:BitBoard, box:Box) -> bool:
    bc, br = box

    h_bits, v_bits = board
    h1 = (h_bits >> h_index(bc, br)) & 1
    h2 = (h_bits >> h_index(bc, br + 1)) & 1
    v1 = (v_bits >> v_index(bc, br)) & 1
    v2 = (v_bits >> v_index(bc + 1, br)) & 1

    return (h1 & h2 & v1 & v2) == 1

def bit_count_completed_boxes(board:BitBoard) -> int:
    cnt = 0
    for br in range(N_BOX):
        for bc in range(N_BOX):
            if bit_is_box_complete(board, box=(bc, br)):
                cnt += 1
    return cnt

def bit_is_edge_claimed(board:BitBoard, edge:Edge) -> bool:
    c, r, d = edge
    if d == H: return ((board[0] >> h_index(c, r)) & 1) == 1
    else:      return ((board[1] >> v_index(c, r)) & 1) == 1

def encode_board(board:Board) -> BitBoard:
    """
        edges를 받아 int 2개로 압축한다.
        BitMask 사용
    """
    h, v = 0, 0
    for r in range(N):
        for c in range(N):
            #print(j, edges[j][i])
            if c != N-1 and board[c][r][H]: 
                h |= 1 << h_index(c, r)
            if r != N-1 and board[c][r][V]:
                v |= 1 << v_index(c, r)
                
    return (h,v)


def decode_bitboard(bit_board:BitBoard) -> Board:
    """
    """
    h, v = bit_board

    board = [[[0 for _ in range(2)] for _ in range(N)] for _ in range(N)]

    for r in range(N):
        for c in range(N):
            if c != N-1: 
                edges[c][r][H] = ((h >> h_index(c, r)) & 1) == 1
            if r != N-1: 
                edges[c][r][V] = ((v >> v_index(c, r)) & 1) == 1

    return edges

def bit_get_legal_actions(bit_board:BitBoard) -> List[Action]:
    """
    bit_board = (h, v)
    반환: [[c, r, d], ...]  (d: 0=H, 1=V)
    """
    h_bits, v_bits = bit_board

    # 유효 비트 길이로 마스킹 (초과 비트가 켜져 있어도 안전)
    H_COUNT = N * (N - 1)   # 수평 엣지 개수
    V_COUNT = (N - 1) * N   # 수직 엣지 개수
    h_bits &= (1 << H_COUNT) - 1
    v_bits &= (1 << V_COUNT) - 1

    actions = []

    # 수평 엣지들: r ∈ [0..n-1], c ∈ [0..n-2]
    # 각 행마다 (n-1)비트를 묶어서 읽으면 인덱스 계산 실수를 줄일 수 있음
    for r in range(N):
        row_bits = (h_bits >> (r * (N - 1))) & ((1 << (N - 1)) - 1)
        for c in range(N - 1):
            if ((row_bits >> c) & 1) == 0:     # 아직 미설치
                actions.append([c, r, 0])      # d=0 (H)

    # 수직 엣지들: r ∈ [0..n-2], c ∈ [0..n-1]
    for r in range(N - 1):
        row_bits = (v_bits >> (r * N)) & ((1 << N) - 1)
        for c in range(N):
            if ((row_bits >> c) & 1) == 0:     # 아직 미설치
                actions.append([c, r, 1])      # d=1 (V)

    return actions

def bit_count_box_edges(board:Board, box:Box) -> int:
    h_bits, v_bits = board
    bc, br = box

    cnt = 0
    # H(br,bc), H(br+1,bc)
    if (h_bits >> h_index(bc, br)) & 1:       cnt += 1
    if (h_bits >> h_index(bc, br + 1)) & 1:   cnt += 1
    # V(br,bc), V(br,bc+1)
    if (v_bits >> v_index(bc, br)) & 1:       cnt += 1
    if (v_bits >> v_index(bc + 1, br)) & 1:   cnt += 1
    return cnt

def bit_makes_third_edge(board: Board, action: Action) -> bool:
    """
    액션이 인접 박스 중 '3번째 엣지'를 만들어서 상대에게 4번째를 헌납할 위험인지 체크.
    """
    c, r, d = action
    for box in get_boxes_adjacent_to_edge(action):
        if bit_count_box_edges(board=board, box=box) == 2:
            # 지금 두면 3이 됨 (위험수)
            return True
    return False

    
def _test_get_available_actions_encoded():
    for i in range(10000):
        h = random.randint(0, 1 << 30)
        v = random.randint(0, 1 << 30)

        encoded_edges = [h, v]
        actions = bit_get_legal_actions(encoded_edges=encoded_edges)

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

        decoded_edges = decode_bitboard(encoded_edges)

        for z in range(2):
            for r in range(6):
                for c in range(6):
                    print(decode_bitboard[c][r][z], end = " ")
                print()
            print("=====")
            

        h_comp, v_comp = encode_board(actions_edges)
        
        assert ((h_comp & h) == 0) and ((v_comp & v) == 0)



def _test_encode_decode():
    t_arr = []

    for i in range(100):
        start = time.time()

        h = random.randint(0, 1 << 30)
        v = random.randint(0, 1 << 30)

        edges = decode_bitboard([h,v])
        h_, v_ = encode_board(edges)

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
    print(decode_bitboard(edges), h_index(1, 3))

    # test Edge Encode, Decode
    _test_encode_decode()
    _test_get_available_actions_encoded()
        