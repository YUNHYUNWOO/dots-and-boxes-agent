from typing import Tuple, List, Dict, Optional
from Util import (
    N_BOX,
    N,
    H_COUNT,
    V_COUNT,
    TOTAL_BOXES,
    H, V,
    h_index,
    v_index,
    check_bounds,
    boxes_adjacent_to_edge,
    is_box_complete,
    count_completed_boxes,
    edge_is_claimed,
    edges_adjacent_to_box,
    encode_Edges,
    decode_Edges
)
# from DotsAndBoxes import DotsAndBoxesEngine
# Board State는 엔진처럼 bitmask로 처리

# 박스 ID를 하나의 정수로 표현 (r * N_BOX + c)
def box_id(c, r):
    return r * N_BOX + c


EDGES = List[int]
dc = [ 0, 1, 0, -1]
dr = [-1, 0, 1,  0]


def init_box_data(edges: List):

    # returns these three
    adj = { box_id(c, r): [] for r in range(N_BOX) for c in range(N_BOX) }
    external_open = { box_id(c, r): 0 for r in range(N_BOX) for c in range(N_BOX) }
    is_candidate = { box_id(c, r): False for r in range(N_BOX) for c in range(N_BOX) }

    for r in range(N_BOX):
        for c in range(N_BOX):
            # 이 박스의 열린 변 개수 세기
            open_dirs = []
            
            for d, adj_edge in enumerate(edges_adjacent_to_box(c, r)):
                if not edge_is_claimed(edges, adj_edge[0], adj_edge[1], adj_edge[2]):  # 선이 안 그려져 있음 = open
                    open_dirs.append(d)

            #print(f'open_dir: {(r,c)}, {open_dirs}, length: {len(open_dirs)}')

            len_open = len(open_dirs)
            if len_open == 0:
                # 이미 완성된 박스 -> 체인/루프 후보 아님
                continue

            # 엔드게임 쪽 체인분해는 보통 '1개 또는 2개의 열린 변'만을 다룬다.
            # (3개 이상이면 아직 미들게임 safe 영역)
            # 필요하면 여기서 필터링:

            if not(1 < len_open < 4): continue

            is_candidate[box_id(c, r)] = True

            for d in open_dirs:
                nc = c + dc[d]
                nr = r + dr[d]

                if 0 <= nc < N_BOX and 0 <= nr < N_BOX:
                    # 이웃 박스와 공유하는 변이 열려 있음 → 내부 연결
                    # 이웃도 후보이든 아니든, 일단 edge는 만들고 나중에 필터링해도 됨
                    adj[box_id(c, r)].append(box_id(nc, nr))
                else:
                    # 보드 바깥과 연결된 열린 변
                    external_open[box_id(c, r)] += 1
    
    return adj, external_open, is_candidate

def get_connected_Components(adj, is_candidate):
    """
        Todo: Juction, 3거리의 체인과 비스무리한건 휴리스틱에서 걸러줘야함.
    """

    visited = { box_id(c,r): False for r in range(N_BOX) for c in range(N_BOX) }
    components = []  # 각 컴포넌트는 [box_id1, box_id2, ...] 리스트

    for r in range(N_BOX):
        for c in range(N_BOX):
            u = box_id(c,r)
            if not is_candidate[u]:
                continue
            if visited[u]:
                continue

            # BFS/DFS 시작
            stack = [u]
            visited[u] = True
            comp = []

            while stack:
                x = stack.pop()
                # print(f'{int(x / N_BOX), x % N_BOX}')
                comp.append(x)
                for y in adj[x]:
                    # print(f"    ->{int(y / N_BOX), y % N_BOX}")

                    # y도 후보여야 "체인/루프"의 일부로 본다
                    if not is_candidate.get(y, False):
                        continue
                    if not visited[y]:
                        visited[y] = True
                        stack.append(y)

            components.append(comp)
    return components

def classify_component(comps, adj):

    res = []
    for comp in comps:
        # comp: 박스 id 리스트
        # 결과: ("chain" or "loop" or "complex", length)

        # 우선 각 노드의 degree 계산
        deg = {}
        for u in comp:
            internal_deg = 0
            for v in adj[u]:
                if v in comp:      # 같은 컴포넌트 내부와의 연결만 센다
                    internal_deg += 1
            deg[u] = internal_deg

        # degree별 카운트
        num_deg1 = sum(1 for u in comp if deg[u] == 1)
        num_other = [u for u in comp if deg[u] not in (1, 2)]

        # 분류 규칙

        # 1) 루프(loop):
        #   - 모든 박스가 degree == 2
        #   - 외부와 열린 변이 없다 (external_open == 0 포함됨)
        is_loop = (num_deg1 == 0 and len(num_other) == 0)
        if is_loop:
            res.append({
                'type': 'loop',
                'length': len(comp)
            })

        # 2) 체인(chain):
        #   - degree == 1 인 박스가 정확히 2개 (양 끝)
        #   - 나머지는 degree == 2
        elif num_deg1 == 2 and len(num_other) == 0:
            res.append({
                'type': 'chain',
                'length': len(comp)
            }) 

        else:
            # 3) 그 외는 복잡한 irregular 구조 (미들게임에서 나올 수 있음)
            res.append({
                'type': 'complex',
                'length': len(comp)
            }) 

    return res


def get_cv(comps):
    def get_fcv(comps):
        fcv = 0
        for comp in comps:
            if (comp['type'] == 'chain') and comp['length'] >= 3:
                fcv += comp['length'] - 4
        
            if (comp['type'] == 'loop') or (comp['type'] == 'complex'):
                fcv += comp['length'] - 8

        return fcv

    def get_tb(comps):
        tb = 0
        for comp in comps:
            if (comp['type'] == 'chain') and comp['length'] >= 3:
                tb = 4
        
            if (comp['type'] == 'loop') or (comp['type'] == 'complex'):
                tb = 8
                break
        return tb
    
    cv = get_fcv(comps) + get_tb(comps)
    return cv




def main():
    test_data = [
        [[[1, 1, 1, 1, 1, 0], 
          [0, 0, 0, 1, 1, 0], 
          [1, 0, 0, 0, 0, 0], 
          [0, 0, 0, 0, 0, 0], 
          [0, 0, 0, 0, 0, 0], 
          [0, 0, 0, 1, 1, 0]], 
          [[1, 0, 1, 0, 1, 1],
           [0, 1, 1, 1, 0, 1],
           [1, 0, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 0, 1],
           [0, 0, 0, 0, 0, 0]]
        ]
    ]
    
    # d, r, c -> c, r, d
    def transpose(edges):
        t_edges = [[[0 for _ in range(2)] for _ in range(N)] for _ in range(N)]

        for r in range(N):
            for c in range(N):
                for d in range(2):
                    t_edges[c][r][d] = edges[d][r][c]
        return t_edges
    
    edges = transpose(test_data[0])
    print(len(edges), len(edges[0]), len(edges[0][0]))

                
    
    print(len(test_data[0]), len(test_data[0][0]), len(test_data[0][0][0]))
    edges = encode_Edges(transpose(test_data[0]))

    adj, external_open, is_candidate = init_box_data(edges)
    print(is_candidate)
    components = get_connected_Components(adj, is_candidate)
    print(components)
    print(classify_component(components, adj))

if __name__ == '__main__':
    main()