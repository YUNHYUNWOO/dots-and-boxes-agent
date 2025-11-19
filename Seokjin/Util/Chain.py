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
    adj = { (r * N_BOX + c): [] for r in range(N_BOX) for c in range(N_BOX) }
    external_open = { (r * N_BOX + c): 0 for r in range(N_BOX) for c in range(N_BOX) }
    is_candidate = { (r * N_BOX + c): False for r in range(N_BOX) for c in range(N_BOX) }

    h_edge, v_edge = edges[0], edges[1]
    cnt_safe_box = 0

    for r in range(N_BOX):
        for c in range(N_BOX):
            # 이 박스의 열린 변 개수 세기
            open_dirs = []
            
            # for d, adj_edge in enumerate(edges_adjacent_to_box(c, r)):
            #     if not edge_is_claimed(edges[0], edges[1], adj_edge[0], adj_edge[1], adj_edge[2]):  # 선이 안 그려져 있음 = open
            #         open_dirs.append(d)

            if not ((h_edge >> (r * (N - 1) + c)) & 1):  # 선이 안 그려져 있음 = open
                open_dirs.append(0)
            if not ((v_edge >> (r * N + c + 1)) & 1):  # 선이 안 그려져 있음 = open
                open_dirs.append(1)
            if not ((h_edge >> ((r + 1) * (N - 1) + c)) & 1):  # 선이 안 그려져 있음 = open
                open_dirs.append(2)
            if not ((v_edge >> (r * N + c)) & 1):  # 선이 안 그려져 있음 = open
                open_dirs.append(3)

            #print(f'open_dir: {(r,c)}, {open_dirs}, length: {len(open_dirs)}')

            if len(open_dirs) == 3:
                cnt_safe_box += 1

            if len(open_dirs) == 0:
                # 이미 완성된 박스 -> 체인/루프 후보 아님
                continue

            # 엔드게임 쪽 체인분해는 보통 '1개 또는 2개의 열린 변'만을 다룬다.
            # (3개 이상이면 아직 미들게임 safe 영역)
            # 필요하면 여기서 필터링:

            if len(open_dirs) != 2: continue

            is_candidate[(r * N_BOX + c)] = True

            for d in open_dirs:
                nc = c + dc[d]
                nr = r + dr[d]

                if 0 <= nc < N_BOX and 0 <= nr < N_BOX:
                    # 이웃 박스와 공유하는 변이 열려 있음 → 내부 연결
                    # 이웃도 후보이든 아니든, 일단 edge는 만들고 나중에 필터링해도 됨
                    adj[(r * N_BOX + c)].append((nr * N_BOX + nc))
                else:
                    # 보드 바깥과 연결된 열린 변
                    external_open[(r * N_BOX + c)] += 1
    
    return adj, external_open, is_candidate, cnt_safe_box

def get_connected_Components(adj, is_candidate):
    """
        Todo: Juction, 3거리의 체인과 비스무리한건 휴리스틱에서 걸러줘야함.
    """
    visited = { (r * N_BOX + c): False for r in range(N_BOX) for c in range(N_BOX) }
    components = []  # 각 컴포넌트는 [box_id1, box_id2, ...] 리스트

    for r in range(N_BOX):
        for c in range(N_BOX):
            u = (r * N_BOX + c)
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
            res.append((0, len(comp))) 

        # 2) 체인(chain):
        #   - degree == 1 인 박스가 정확히 2개 (양 끝)
        #   - 나머지는 degree == 2
        if num_deg1 == 2 and len(num_other) == 0:
            res.append((1, len(comp))) 

        # 3) 그 외는 복잡한 irregular 구조 (미들게임에서 나올 수 있음)
        # res.append({
        #     'type': 'complex',
        #     'length': len(comp)
        # }) 
    return res

# def get_cv(comps, cnt_safe_box):
#     cv = 0
#     tb = 4
#     cnt = 0
#     for comp in comps:
#         if (comp[0] == 1) and comp[1] >= 3: # chain
#             cv += comp[1] - 4
#             cnt += comp[1]
#         elif (comp[0] == 0): # loop
#             cv += comp[1] - 8
#             cnt += comp[1]
#             tb = 8
#     cv += tb

#     if (cnt_safe_box + cnt) < 25: # 3변 이상 칠해진 상자 존재, 
#         cv -= 4 * max(1, cnt_safe_box)
#     return cv

def get_cv(comps):
    cv = 0
    tb = 4
    cnt = 0
    for comp in comps:
        if (comp[0] == 1) and comp[1] >= 3: # chain
            cv += comp[1] - 4
            cnt += comp[1]
        elif (comp[0] == 0): # loop
            cv += comp[1] - 8
            cnt += comp[1]
            tb = 8
    cv += tb

    if (cnt_safe_box + cnt) < 25: # 3변 이상 칠해진 상자 존재, 
        cv -= 4 * max(1, cnt_safe_box)
    return cv

# ----- 체인/루프 컴포넌트 정보 확장 분석 -----

def _build_component_info(components, adj):
    """
    components: get_connected_Components 가 돌려준 [ [box_id, ...], ... ]
    adj       : init_box_data 가 돌려준 인접 리스트

    반환:
      comp_info: 각 컴포넌트에 대한 정보 리스트
        [{
            'type': 'chain' | 'loop' | 'complex',
            'length': int,
            'boxes': set(box_id, ...),  # 이 컴포넌트에 속한 박스들
            'deg': { box_id: internal_degree, ... }
         }, ...]
      box_to_comp: { box_id: comp_index }
      deg_map    : { box_id: internal_degree }  (빠른 조회용)
    """
    comp_info = []
    box_to_comp = {}
    deg_map = {}

    for idx, comp in enumerate(components):
        comp_set = set(comp)
        deg = {}

        # 내부 degree 계산
        for u in comp:
            d = 0
            for v in adj[u]:
                if v in comp_set:
                    d += 1
            deg[u] = d
            deg_map[u] = d
            box_to_comp[u] = idx

        num_deg1 = sum(1 for u in comp if deg[u] == 1)
        num_other = sum(1 for u in comp if deg[u] not in (1, 2))

        # 분류 규칙:
        #  - loop : 모든 박스 degree == 2
        #  - chain: degree==1 인 박스가 정확히 2개 (양 끝), 나머지는 2
        #           + 길이 1짜리(고립된 1박스)도 편의상 chain 취급
        if num_deg1 == 0 and num_other == 0:
            ctype = 'loop'
        elif (num_deg1 == 2 and num_other == 0) or (len(comp) == 1):
            ctype = 'chain'
        else:
            ctype = 'complex'

        comp_info.append({
            'type': ctype,
            'length': len(comp),
            'boxes': comp_set,
            'deg': deg
        })

    return comp_info, box_to_comp, deg_map

# ----- 특정 엣지가 체인/루프를 여는 수인지 판별 -----

def is_opening_edge(edges, o, r, c):
    """
    주어진 보드 상태(edges)에서, 엣지 (o, r, c)가
    '체인 또는 루프를 여는 수'인지 판별한다.

    인자:
      edges : encode_Edges 로 인코딩된 현재 보드의 엣지 비트마스크
      o     : H 또는 V (Util 에서 import 한 상수)
      r, c  : 엣지의 인덱스

    반환:
      (is_opening, info)

      is_opening: bool
      info: None 또는
        [
          {
            'type': 'chain' | 'loop',      # 어떤 종류의 컴포넌트를 여는지
            'component_index': int,        # components 리스트에서의 인덱스
            'component_length': int        # 체인/루프 길이(박스 개수)
          },
          ...
        ]

      보통 한 엣지가 여러 컴포넌트에 동시에 속하지 않으므로
      info 리스트는 길이 0 또는 1 일 것이다.
    """

    # 현재 보드에서 체인/루프 후보 박스 정보 구축
    adj, external_open, is_candidate = init_box_data(edges)
    components = get_connected_Components(adj, is_candidate)

    if not components:
        return False, None

    comp_info, box_to_comp, deg_map = _build_component_info(components, adj)

    # 이 엣지에 인접한 박스들 찾기 (최대 2개)
    adj_boxes = boxes_adjacent_to_edge(o, r, c)

    opening_details = []

    # 인접한 박스들 중에서, 체인/루프 컴포넌트를 여는지 검사
    for (bc, br) in adj_boxes:
        if not (0 <= bc < N_BOX and 0 <= br < N_BOX):
            continue

        bid = box_id(bc, br)

        # 체인/루프 후보 박스가 아니면 무시
        if not is_candidate.get(bid, False):
            continue

        comp_idx = box_to_comp.get(bid)
        if comp_idx is None:
            continue

        info = comp_info[comp_idx]
        ctype = info['type']

        # 1) 루프: 루프 컴포넌트에 붙은 어떤 열린 엣지를 그리면
        #    그 루프를 깨는(여는) 수가 된다고 본다.
        if ctype == 'loop':
            opening_details.append({
                'type': 'loop',
                'component_index': comp_idx,
                'component_length': info['length']
            })
            continue

        # 2) 체인: 체인의 '끝점' 박스에 붙은 외부 엣지를 그릴 때
        #    그 체인을 여는 수가 된다.
        if ctype == 'chain':
            comp_boxes = info['boxes']
            deg = info['deg']

            # 이 엣지에 인접한 박스들 중, 이 체인 컴포넌트에 속한 것들
            boxes_in_comp_adjacent = []
            for (bc2, br2) in adj_boxes:
                if not (0 <= bc2 < N_BOX and 0 <= br2 < N_BOX):
                    continue
                bid2 = box_id(bc2, br2)
                if bid2 in comp_boxes:
                    boxes_in_comp_adjacent.append(bid2)

            # 체인 내부의 '연결 엣지'라면 두 박스(양쪽)가 같은 체인에 속한다.
            # 체인을 여는 '입구 엣지'는 체인 박스 하나에만 붙는다.
            if len(boxes_in_comp_adjacent) != 1:
                continue

            endpoint_bid = boxes_in_comp_adjacent[0]

            # 체인의 끝점이어야 한다 (degree 1) 또는 길이1 체인(고립된 박스)은 degree 0
            if deg_map.get(endpoint_bid, 0) > 1:
                # 내부 박스라면 이 엣지는 체인 내부를 잇는 엣지 → 여는 수 아님
                continue

            # 이 엣지가 진짜로 그 박스의 "열린 변"인지 확인
            # (안 그려진 edge 이고, 그 박스에 인접한 엣지 중 하나여야 함)
            c_box = endpoint_bid % N_BOX
            r_box = endpoint_bid // N_BOX
            is_open_of_box = False
            for o2, rr2, cc2 in edges_adjacent_to_box(c_box, r_box):
                if not edge_is_claimed(edges[0], edges[1], o2, rr2, cc2):
                    if o2 == o and rr2 == r and cc2 == c:
                        is_open_of_box = True
                        break

            if not is_open_of_box:
                continue

            opening_details.append({
                'type': 'chain',
                'component_index': comp_idx,
                'component_length': info['length']
            })

    if not opening_details:
        return False, None

    # 중복 제거 (같은 컴포넌트가 여러 번 추가됐을 수 있음)
    unique = {}
    for d in opening_details:
        key = (d['type'], d['component_index'])
        if key not in unique:
            unique[key] = d

    return True, list(unique.values())

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
    print(classify_component(components, adj))

    print(is_opening_edge(edges, 0, 0, 1))

if __name__ == '__main__':
    main()