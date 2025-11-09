from typing import List, Tuple, Dict, Optional
import numpy as np
import random
from .BasePolicy import BasePolicy
# ---------- Helpers to work with the Env observation ----------

def _edges_drawn_from_obs(observation: Dict[str, np.ndarray]):
    """
    Convert observation's h_edges/v_edges (values in {-1,0,1}) into boolean arrays:
    True = edge already drawn by someone, False = empty.
    """
    h = np.array(observation["h_edges"])
    v = np.array(observation["v_edges"])
    h_drawn = (h != 0)
    v_drawn = (v != 0)
    return h_drawn, v_drawn

def _box_edge_counts(h_drawn: np.ndarray, v_drawn: np.ndarray) -> np.ndarray:
    """
    For each box (r,c), count how many of its 4 edges are drawn.
    h_drawn: shape = (n_box+1, n_box)
    v_drawn: shape = (n_box, n_box+1)
    Returns: counts shape (n_box, n_box)
    """
    n_box = h_drawn.shape[1]  # since h_drawn is (n_box+1, n_box)
    counts = np.zeros((n_box, n_box), dtype=int)
    for r in range(n_box):
        for c in range(n_box):
            cnt = 0
            if h_drawn[r, c]:     cnt += 1  # top
            if h_drawn[r+1, c]:   cnt += 1  # bottom
            if v_drawn[r, c]:     cnt += 1  # left
            if v_drawn[r, c+1]:   cnt += 1  # right
            counts[r, c] = cnt
    return counts

def _edge_is_undrawn(h_drawn, v_drawn, edge):
    ori, r, c = edge  # ori: 0=H, 1=V (Env convention)
    if ori == 0:
        return not h_drawn[r, c]
    else:
        return not v_drawn[r, c]


# ---------- Safe-move detection ----------

def list_safe_moves(observation: Dict[str, np.ndarray]) -> List[Tuple[int,int,int]]:
    """
    Safe move = 이 수를 두었을 때
      - 어떤 박스도 3변(=count 2 -> 3)이 되지 않고
      - 어떤 박스도 완성(=count 3 -> 4)되지 않는 수.
    즉, 인접 박스의 현재 edge-count가 모두 {0,1}인 엣지만 안전수로 인정.
    """
    h_drawn, v_drawn = _edges_drawn_from_obs(observation)
    n_box = h_drawn.shape[1]
    counts = _box_edge_counts(h_drawn, v_drawn)

    safe = []

    # Horizontal edges
    for r in range(n_box + 1):
        for c in range(n_box):
            if not h_drawn[r, c]:
                # 인접 박스: (r-1,c) [위], (r,c) [아래]
                adj = []
                if 0 <= r - 1 < n_box: adj.append(counts[r-1, c])
                if 0 <= r     < n_box: adj.append(counts[r,   c])

                # 어떤 인접 박스라도 count >= 2 이면 unsafe
                if all(cnt <= 1 for cnt in adj):
                    safe.append((0, r, c))

    # Vertical edges
    for r in range(n_box):
        for c in range(n_box + 1):
            if not v_drawn[r, c]:
                # 인접 박스: (r,c-1) [왼], (r,c) [오]
                adj = []
                if 0 <= c - 1 < n_box: adj.append(counts[r, c-1])
                if 0 <= c     < n_box: adj.append(counts[r, c])

                if all(cnt <= 1 for cnt in adj):
                    safe.append((1, r, c))

    return safe



# ---------- Component (Chain/Loop) decomposition ----------

# Represent a box by its (r,c) in [0..n_box-1]x[0..n_box-1].
# We'll build a graph where nodes are boxes; we connect two adjacent boxes if their shared edge is UNDRAWN.
# Boundary UNDRAWN edges connect the box to an implicit OUTSIDE node, which we denote by None.

def _adjacent_boxes_and_edges(n_box: int, h_drawn, v_drawn):
    """
    Build adjacency for UNDRAWN shared edges.
    Returns:
        adj: dict[(r,c)] -> set of neighbors (including None for outside)
        edge_between: dict[((u),(v))] -> list of edge tuples (ori,r,c) that connect them (usually length 1)
    """
    adj: Dict[Tuple[int,int], set] = {}
    edge_between: Dict[Tuple[Optional[Tuple[int,int]], Optional[Tuple[int,int]]], List[Tuple[int,int,int]]] = {}

    def add_edge(a, b, edge):
        if a not in adj: adj[a] = set()
        if b not in adj: adj[b] = set()
        adj[a].add(b)
        adj[b].add(a)
        key = (a, b) if (a is None or b is None or a <= b) else (b, a)
        edge_between.setdefault(key, []).append(edge)

    # init nodes
    for r in range(n_box):
        for c in range(n_box):
            adj[(r,c)] = set()

    # internal shared edges
    for r in range(n_box):
        for c in range(n_box):
            # Right neighbor share vertical edge v_drawn[r, c+1]
            if c+1 < n_box and not v_drawn[r, c+1]:
                a = (r, c)
                b = (r, c+1)
                edge = (1, r, c+1)  # vertical (r, c+1)
                add_edge(a, b, edge)
            # Bottom neighbor share horizontal edge h_drawn[r+1, c]
            if r+1 < n_box and not h_drawn[r+1, c]:
                a = (r, c)
                b = (r+1, c)
                edge = (0, r+1, c)  # horizontal (r+1, c)
                add_edge(a, b, edge)

    # boundary UNDRAWN edges connect to OUTSIDE None
    for r in range(n_box):
        for c in range(n_box):
            # top boundary
            if not h_drawn[r, c]:
                add_edge((r, c), None, (0, r, c))
            # bottom boundary
            if not h_drawn[r+1, c]:
                add_edge((r, c), None, (0, r+1, c))
            # left boundary
            if not v_drawn[r, c]:
                add_edge((r, c), None, (1, r, c))
            # right boundary
            if not v_drawn[r, c+1]:
                add_edge((r, c), None, (1, r, c+1))

    return adj, edge_between

def decompose_components(observation: Dict[str, np.ndarray]):
    """
    Decompose the current UNDRAWN-edge structure into components.
    Returns a list of components:
      {
        "type": "chain" or "loop",
        "boxes": set of (r,c),
        "length": int,   # number of boxes in component
        "open_edges_to_outside": list of (ori,r,c),  # edges that connect to outside (for chains)
        "internal_undrawn_edges": list of (ori,r,c), # undrawn edges strictly between boxes (for loops and chains)
      }
    """
    h_drawn, v_drawn = _edges_drawn_from_obs(observation)
    n_box = h_drawn.shape[1]

    adj, edge_between = _adjacent_boxes_and_edges(n_box, h_drawn, v_drawn)

    visited = set()
    components = []

    # gather all boxes that are part of any UNDRAWN-edge adjacency (degree>0)
    candidates = [b for b, neigh in adj.items() if b is not None and len(neigh) > 0]

    for start in candidates:
        if start in visited:
            continue
        # BFS over boxes only; track outside presence
        q = [start]
        visited.add(start)
        comp_boxes = {start}
        touches_outside = False
        degrees_inside = {}  # degree excluding outside
        inside_edges = set()
        open_to_outside_edges = set()

        while q:
            u = q.pop()
            for v in adj[u]:
                key = (u, v) if (u is None or v is None or u <= v) else (v, u)
                edges_uv = edge_between.get(key, [])
                if v is None:
                    touches_outside = True
                    for e in edges_uv:
                        open_to_outside_edges.add(e)
                else:
                    # internal neighbor
                    for e in edges_uv:
                        inside_edges.add(e)
                    # accumulate degree (inside only)
                    degrees_inside[u] = degrees_inside.get(u, 0) + 1
                    degrees_inside[v] = degrees_inside.get(v, 0) + 1
                    if v not in visited:
                        visited.add(v)
                        comp_boxes.add(v)
                        q.append(v)

        if not comp_boxes:
            continue

        length = len(comp_boxes)

        if touches_outside:
            ctype = "chain"
        else:
            # in a loop, every box has degree 2 internally
            all_deg_two = all(degrees_inside.get(b, 0) == 2 for b in comp_boxes)
            ctype = "loop" if all_deg_two else "chain"  # fallback just in case

        components.append({
            "type": ctype,
            "boxes": comp_boxes,
            "length": length,
            "open_edges_to_outside": list(open_to_outside_edges),
            "internal_undrawn_edges": list(inside_edges),
        })

    # Sort components by a stable rule (e.g., by length asc, loop before chain for same length)
    components.sort(key=lambda x: (x["length"], 0 if x["type"]=="loop" else 1))

    return components


# ---------- Controlled Value & Opening Choice ----------

def controlled_value(components: List[Dict]) -> int:
    """Compute Controlled Value (CV) for the controller."""
    sum_chains = sum(c["length"] - 2 for c in components if c["type"] == "chain")
    sum_loops  = sum(c["length"] - 4 for c in components if c["type"] == "loop")
    tb = 4 if any(c["type"]=="chain" for c in components) else (8 if any(c["type"]=="loop" for c in components) else 0)
    return (sum_chains + sum_loops) - tb

def choose_endgame_opening(components: List[Dict]) -> int:
    """
    Decide which component to OPEN (index in components), using standard rules:
    - If CV >= 2: open the SHORTEST chain; if none, the shortest loop.
    - Else (CV <= 1): open a 3-chain if exists; else the shortest loop; else the shortest chain.
    """
    cv = controlled_value(components)
    chains = [i for i,c in enumerate(components) if c["type"]=="chain"]
    loops  = [i for i,c in enumerate(components) if c["type"]=="loop"]

    if cv >= 2:
        if chains:
            return min(chains, key=lambda i: components[i]["length"])
        else:
            return min(loops, key=lambda i: components[i]["length"])
    else:
        three_chains = [i for i in chains if components[i]["length"] == 3]
        if three_chains:
            return three_chains[0]
        elif loops:
            return min(loops, key=lambda i: components[i]["length"])
        elif chains:
            return min(chains, key=lambda i: components[i]["length"])
        else:
            raise RuntimeError("No components found to open.")


# ---------- Translate a chosen component into a concrete (ori,r,c) move ----------

def pick_opening_edge_for_component(observation: Dict[str, np.ndarray], component: Dict) -> Tuple[int,int,int]:
    """
    For a chain: pick any open edge to OUTSIDE (end-edge). Prefer one that's valid according to action mask externally.
    For a loop: pick any internal UNDRAWN edge between two boxes.
    """
    if component["type"] == "chain":
        # Prefer outside edges (end caps)
        if not component["open_edges_to_outside"]:
            # Rare fallback: if graph classified it as chain but no outside edges were recorded,
            # just use any internal undrawn edge.
            return component["internal_undrawn_edges"][0]
        return component["open_edges_to_outside"][0]
    else:  # loop
        # Any internal undrawn edge will open the loop
        return component["internal_undrawn_edges"][0]


# ---------- Policy that uses the endgame logic ----------

class EndgamePolicy(BasePolicy):
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def get_action(self, observation: Dict[str, np.ndarray], info: Dict, env) -> Tuple[int,int,int]:
        mask = info["action_mask"]

        # 1) 안전수 중에서 랜덤으로 선택
        safe = list_safe_moves(observation)
        if safe:
            candidates = [(ori, r, c) for (ori, r, c) in safe if not mask[ori, r, c]]
            if candidates:
                return self.rng.choice(candidates)
            # 안전수는 있는데 전부 마스크되어 있으면 엔드게임 로직으로 폴백

        # 2) 안전수 0 → 체인/루프 분해 + 최적 오프닝
        components = decompose_components(observation)
        if components:
            idx = choose_endgame_opening(components)
            # 우선 후보
            move = pick_opening_edge_for_component(observation, components[idx])
            if not mask[move[0], move[1], move[2]]:
                return move
            # 같은 컴포넌트 내 다른 엣지들로 폴백
            for e in (components[idx]["open_edges_to_outside"] + components[idx]["internal_undrawn_edges"]):
                if not mask[e[0], e[1], e[2]]:
                    return e

        # 3) 최종 폴백: 아무 합법수 랜덤
        n = observation["h_edge"].shape[1]
        all_moves = []
        for ori in (0, 1):
            R = (n + 1) if ori == 0 else n
            C = n if ori == 0 else (n + 1)
            for r in range(R):
                for c in range(C):
                    if not mask[ori, r, c]:
                        all_moves.append((ori, r, c))
        if not all_moves:
            raise RuntimeError("No available moves found (action mask fully blocked).")
        return self.rng.choice(all_moves)
