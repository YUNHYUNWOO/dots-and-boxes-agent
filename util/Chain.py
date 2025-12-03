"""Chain and loop analysis utilities for Dots and Boxes bitboards."""

from typing import Dict, List, Tuple

from config import BitBoard, N, N_BOX
from util.bit_dnb_util import bit_is_edge_claimed
from util.dnb_util import get_boxes_adjacent_to_edge, get_edges_adjacent_to_box

# Direction deltas in (c, r) order for H, V, H, V traversal.
DC = [0, 1, 0, -1]
DR = [-1, 0, 1, 0]


def box_id(c: int, r: int) -> int:
    """Unique id for a box at (c, r)."""

    return r * N_BOX + c


def init_box_data(board: BitBoard) -> Tuple[Dict[int, List[int]], Dict[int, bool]]:
    """Identify candidate boxes (exactly two open edges) and adjacency graph."""

    adjacency: Dict[int, List[int]] = {box_id(c, r): [] for r in range(N_BOX) for c in range(N_BOX)}
    is_candidate: Dict[int, bool] = {box_id(c, r): False for r in range(N_BOX) for c in range(N_BOX)}

    for r in range(N_BOX):
        for c in range(N_BOX):
            open_dirs: List[int] = []
            for d, edge in enumerate(get_edges_adjacent_to_box((c, r))):
                if not bit_is_edge_claimed(board, edge):
                    open_dirs.append(d)

            if len(open_dirs) != 2:
                # Either already closed or more than two openings (safe midgame area).
                continue

            is_candidate[box_id(c, r)] = True
            for d in open_dirs:
                nc = c + DC[d]
                nr = r + DR[d]
                if 0 <= nc < N_BOX and 0 <= nr < N_BOX:
                    adjacency[box_id(c, r)].append(box_id(nc, nr))

    return adjacency, is_candidate

def get_connected_components(adj: Dict[int, List[int]], is_candidate: Dict[int, bool]) -> List[List[int]]:
    """Return connected components among candidate boxes."""

    visited: Dict[int, bool] = {box_id(c, r): False for r in range(N_BOX) for c in range(N_BOX)}
    components: List[List[int]] = []

    for r in range(N_BOX):
        for c in range(N_BOX):
            u = box_id(c, r)
            if not is_candidate.get(u, False) or visited[u]:
                continue

            stack = [u]
            visited[u] = True
            comp: List[int] = []

            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj[x]:
                    if not is_candidate.get(y, False) or visited[y]:
                        continue
                    visited[y] = True
                    stack.append(y)

            components.append(comp)
    return components


def classify_component(comps: List[List[int]], adj: Dict[int, List[int]]) -> List[Dict[str, int | str]]:
    """Classify components into chain/loop/complex with their lengths."""

    res: List[Dict[str, int | str]] = []
    for comp in comps:
        deg: Dict[int, int] = {}
        for u in comp:
            deg[u] = sum(1 for v in adj[u] if v in comp)

        num_deg1 = sum(1 for u in comp if deg[u] == 1)
        num_other = [u for u in comp if deg[u] not in (1, 2)]

        if num_deg1 == 0 and len(num_other) == 0:
            res.append({"type": "loop", "length": len(comp)})
        elif num_deg1 == 2 and len(num_other) == 0:
            res.append({"type": "chain", "length": len(comp)})
        else:
            res.append({"type": "complex", "length": len(comp)})

    return res


def get_cv(comps: List[Dict[str, int | str]]) -> int:
    """Compute control value (CV) metric from component list."""

    def get_fcv(items: List[Dict[str, int | str]]) -> int:
        fcv = 0
        for comp in items:
            length = int(comp["length"])
            if comp["type"] == "chain" and length >= 3:
                fcv += length - 4
            if comp["type"] in ("loop", "complex"):
                fcv += length - 8
        return fcv

    def get_tb(items: List[Dict[str, int | str]]) -> int:
        tb = 0
        for comp in items:
            length = int(comp["length"])
            if comp["type"] == "chain" and length >= 3:
                tb = 4
            if comp["type"] in ("loop", "complex"):
                tb = 8
                break
        return tb

    return get_fcv(comps) + get_tb(comps)


def get_chain_risk(comps: List[Dict[str, int | str]]) -> int:
    """Return cumulative length penalty for long chains/loops."""

    cv = 0
    for comp in comps:
        length = int(comp["length"])
        if comp["type"] == "chain" and length >= 3:
            cv += length - 2
        elif comp["type"] == "loop":
            cv += length - 4
    return cv

def main():
    from .bit_dnb_util import encode_board
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
        ],
        [[[0, 0, 0, 0, 0, 0], 
          [0, 0, 0, 0, 0, 0], 
          [0, 0, 0, 0, 0, 0], 
          [1, 1, 1, 1, 1, 1], 
          [0, 0, 0, 0, 0, 0], 
          [0, 0, 0, 0, 0, 0]], 
          [[0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]]
        ]
    ]
    
    def transpose(edges):
        t_edges = [[[0 for _ in range(2)] for _ in range(N)] for _ in range(N)]

        for r in range(N):
            for c in range(N):
                for d in range(2):
                    t_edges[c][r][d] = edges[d][r][c]
        return t_edges

    for data in test_data:
        # d, r, c -> c, r, d

        edges = transpose(data)
        print(len(edges), len(edges[0]), len(edges[0][0]))
        
        print(len(test_data[0]), len(test_data[0][0]), len(test_data[0][0][0]))
        edges = encode_board(edges)

        adj, is_candidate = init_box_data(edges)
        print(is_candidate)
        components = get_connected_components(adj, is_candidate)
        print(components)
        print(classify_component(components, adj))

if __name__ == '__main__':
    main()
