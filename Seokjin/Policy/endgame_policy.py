from typing import List, Tuple, Dict, Optional
import numpy as np
import random

"""
Board size assumption
- Boxes: 5x5
- Points: 6x6
- Observation:
    observation["edges"]: shape (6,6,2), dtype=bool
        edges[r, c, 0] = horizontal edge starting at point (r,c) to (r, c+1)
            valid indices: r in [0..5], c in [0..4]
        edges[r, c, 1] = vertical edge starting at point (r,c) to (r+1, c)
            valid indices: r in [0..4], c in [0..5]
    observation["cur_player"]: 0 or 1  (not used by policy logic here)

- Action: tuple (edge_type, row, col)
    edge_type: 0 = horizontal, 1 = vertical
    row, col follow the same valid ranges as above.
- info["action_mask"][edge_type, row, col] is assumed:
    False -> legal (selectable), True -> illegal
"""

N_BOX = 5       # 5x5 boxes
N_PT  = N_BOX+1 # 6x6 points


# ---------- Helpers to work with the new Env observation ----------

def _edges_drawn_from_obs(observation: Dict[str, np.ndarray]):
    """
    Convert observation["edges"] (6,6,2) into:
        h_drawn: (6,5)  bool  (horizontal edges)
        v_drawn: (5,6)  bool  (vertical edges)
    """
    edges = np.asarray(observation["edges"], dtype=bool)
    assert edges.shape == (N_PT, N_PT, 2), f"edges must be {(N_PT, N_PT, 2)}, got {edges.shape}"

    # Horizontal: r:0..5, c:0..4 from edges[:,:,0]
    h_drawn = edges[:, :N_BOX, 0]   # (6,5)

    # Vertical: r:0..4, c:0..5 from edges[:,:,1]
    v_drawn = edges[:N_BOX, :, 1]   # (5,6)

    return h_drawn, v_drawn


def _box_edge_counts(h_drawn: np.ndarray, v_drawn: np.ndarray) -> np.ndarray:
    """
    For each box (r,c) in 0..4 x 0..4, count how many of its 4 edges are drawn.
    h_drawn: (6,5) top/bottom edges
    v_drawn: (5,6) left/right edges
    Returns: counts shape (5,5)
    """
    counts = np.zeros((N_BOX, N_BOX), dtype=int)
    for r in range(N_BOX):
        for c in range(N_BOX):
            cnt = 0
            if h_drawn[r,   c]: cnt += 1  # top
            if h_drawn[r+1, c]: cnt += 1  # bottom
            if v_drawn[r, c]:   cnt += 1  # left
            if v_drawn[r, c+1]: cnt += 1  # right
            counts[r, c] = cnt
    return counts


def _edge_is_undrawn(h_drawn, v_drawn, edge: Tuple[int,int,int]) -> bool:
    et, r, c = edge
    if et == 0:
        # horizontal: r in [0..5], c in [0..4]
        return not h_drawn[r, c]
    else:
        # vertical: r in [0..4], c in [0..5]
        return not v_drawn[r, c]


# ---------- Safe / Capture moves ----------

def list_safe_moves(observation: Dict[str, np.ndarray]) -> List[Tuple[int,int,int]]:
    """
    Safe move = placing this edge does NOT create a 3-edge box (i.e., no adjacent box goes from count<=1 to 2 or more?).
    Classic Dots&Boxes 'safe' commonly means: it doesn't raise any adjacent box to 3 edges and doesn't complete a box.
    Equivalently: all adjacent boxes currently have counts in {0,1}.
    """
    h_drawn, v_drawn = _edges_drawn_from_obs(observation)
    counts = _box_edge_counts(h_drawn, v_drawn)

    safe: List[Tuple[int,int,int]] = []

    # Horizontal candidates
    for r in range(N_PT):
        for c in range(N_BOX):
            if not h_drawn[r, c]:
                adj = []
                if 0 <= r-1 < N_BOX: adj.append(counts[r-1, c])  # box above
                if 0 <= r   < N_BOX: adj.append(counts[r,   c])  # box below
                if all(cnt <= 1 for cnt in adj):
                    safe.append((0, r, c))

    # Vertical candidates
    for r in range(N_BOX):
        for c in range(N_PT):
            if not v_drawn[r, c]:
                adj = []
                if 0 <= c-1 < N_BOX: adj.append(counts[r, c-1])  # left box
                if 0 <= c   < N_BOX: adj.append(counts[r, c])    # right box
                if all(cnt <= 1 for cnt in adj):
                    safe.append((1, r, c))

    return safe


def list_capture_moves(observation: Dict[str, np.ndarray]) -> List[Tuple[int,int,int]]:
    """
    Capture move = this edge completes at least one adjacent box (count 3 -> 4).
    """
    h_drawn, v_drawn = _edges_drawn_from_obs(observation)
    counts = _box_edge_counts(h_drawn, v_drawn)

    caps: List[Tuple[int,int,int]] = []

    # Horizontal
    for r in range(N_PT):
        for c in range(N_BOX):
            if not h_drawn[r, c]:
                adj = []
                if 0 <= r-1 < N_BOX: adj.append(counts[r-1, c])
                if 0 <= r   < N_BOX: adj.append(counts[r,   c])
                if any(cnt == 3 for cnt in adj):
                    caps.append((0, r, c))

    # Vertical
    for r in range(N_BOX):
        for c in range(N_PT):
            if not v_drawn[r, c]:
                adj = []
                if 0 <= c-1 < N_BOX: adj.append(counts[r, c-1])
                if 0 <= c   < N_BOX: adj.append(counts[r, c])
                if any(cnt == 3 for cnt in adj):
                    caps.append((1, r, c))

    return caps


# ---------- Component (Chain/Loop) decomposition ----------

# Represent a box by (r,c) in [0..4]x[0..4].
# Build a graph where nodes are boxes; connect two boxes if their shared edge is UNDRAWN.
# Boundary UNDRAWN edges connect the box to an implicit OUTSIDE node = None.

def _adjacent_boxes_and_edges(h_drawn: np.ndarray, v_drawn: np.ndarray):
    """
    Build adjacency for UNDRAWN shared/boundary edges.
    Returns:
        adj: dict[(r,c)] -> set of neighbors (including None for outside)
        edge_between: dict[((u),(v))] -> list of (edge_type, r, c) that connect them
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
    for r in range(N_BOX):
        for c in range(N_BOX):
            adj[(r,c)] = set()

    # internal UNDRAWN shared edges
    for r in range(N_BOX):
        for c in range(N_BOX):
            # Right neighbor: shared vertical v_drawn[r, c+1]
            if c+1 < N_BOX and not v_drawn[r, c+1]:
                a, b = (r, c), (r, c+1)
                edge = (1, r, c+1)  # vertical
                add_edge(a, b, edge)
            # Bottom neighbor: shared horizontal h_drawn[r+1, c]
            if r+1 < N_BOX and not h_drawn[r+1, c]:
                a, b = (r, c), (r+1, c)
                edge = (0, r+1, c)  # horizontal
                add_edge(a, b, edge)

    # boundary UNDRAWN edges connect to outside None
    for r in range(N_BOX):
        for c in range(N_BOX):
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
    Decompose UNDRAWN-edge regions into components of boxes.
    A component is a chain if it touches outside; otherwise a loop (all internal degree 2).
    """
    h_drawn, v_drawn = _edges_drawn_from_obs(observation)
    adj, edge_between = _adjacent_boxes_and_edges(h_drawn, v_drawn)

    visited = set()
    components = []

    # boxes that participate (degree>0)
    candidates = [b for b, nb in adj.items() if b is not None and len(nb) > 0]

    for start in candidates:
        if start in visited:
            continue

        q = [start]
        visited.add(start)
        comp_boxes = {start}
        touches_outside = False
        degrees_inside: Dict[Tuple[int,int], int] = {}
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
                    for e in edges_uv:
                        inside_edges.add(e)
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
            all_deg_two = all(degrees_inside.get(b, 0) == 2 for b in comp_boxes)
            ctype = "loop" if all_deg_two else "chain"

        components.append({
            "type": ctype,
            "boxes": comp_boxes,
            "length": length,
            "open_edges_to_outside": list(open_to_outside_edges),
            "internal_undrawn_edges": list(inside_edges),
        })

    components.sort(key=lambda x: (x["length"], 0 if x["type"] == "loop" else 1))
    return components


# ---------- Controlled Value & Opening Choice ----------

def controlled_value(components: List[Dict]) -> int:
    """Compute Controlled Value (CV) for the controller."""
    sum_chains = sum(c["length"] - 2 for c in components if c["type"] == "chain")
    sum_loops  = sum(c["length"] - 4 for c in components if c["type"] == "loop")
    tb = 4 if any(c["type"] == "chain" for c in components) else (8 if any(c["type"] == "loop") else 0)
    return (sum_chains + sum_loops) - tb


def choose_endgame_opening(components: List[Dict]) -> int:
    """
    Standard opening rules:
    - If CV >= 2: open the SHORTEST chain; if none, the shortest loop.
    - Else: open a 3-chain if exists; else shortest loop; else shortest chain.
    """
    cv = controlled_value(components)
    chains = [i for i, c in enumerate(components) if c["type"] == "chain"]
    loops  = [i for i, c in enumerate(components) if c["type"] == "loop"]

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
            raise RuntimeError("No components to open.")


# ---------- Opening edge selection ----------

def pick_opening_edge_for_component(component: Dict) -> Tuple[int,int,int]:
    """
    For a chain: pick any edge that touches outside (end-cap).
    For a loop: pick any undrawn internal edge to 'open' the loop.
    """
    if component["type"] == "chain":
        if component["open_edges_to_outside"]:
            return component["open_edges_to_outside"][0]
        # Fallback: if misclassified, pick any internal
        return component["internal_undrawn_edges"][0]
    else:  # loop
        return component["internal_undrawn_edges"][0]


# ---------- Policy ----------

class EndgamePolicy:
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def get_action(self, observation: Dict[str, np.ndarray], info: Dict, env) -> Tuple[int,int,int]:
        mask = info["action_mask"]  # False=legal, True=illegal

        # 0) Punish mistakes first: if any capture exists, take one.
        capture = list_capture_moves(observation)
        if capture:
            cap_candidates = [(o, r, c) for (o, r, c) in capture if not mask[o, r, c]]
            if cap_candidates:
                return self.rng.choice(cap_candidates)

        # 1) Pre-endgame: prefer safe moves
        safe = list_safe_moves(observation)
        if safe:
            candidates = [(o, r, c) for (o, r, c) in safe if not mask[o, r, c]]
            if candidates:
                return self.rng.choice(candidates)
            # Rare case: all safe moves masked -> fallthrough

        # 2) Endgame: decompose into chains/loops and open optimally
        components = decompose_components(observation)
        if components:
            idx = choose_endgame_opening(components)
            move = pick_opening_edge_for_component(components[idx])
            if not mask[move[0], move[1], move[2]]:
                return move
            # Fallback within the same component
            for e in (components[idx]["open_edges_to_outside"] + components[idx]["internal_undrawn_edges"]):
                if not mask[e[0], e[1], e[2]]:
                    return e

        # 3) 최종 폴백: 남은 합법 수 중 랜덤
        pool = []
        # Horizontal edges: (edge_type=0) r in [0..5], c in [0..4]
        for r in range(N_PT):      # N_PT = 6
            for c in range(N_BOX): # N_BOX = 5
                if not mask[0, r, c]:
                    pool.append((0, r, c))
        # Vertical edges: (edge_type=1) r in [0..4], c in [0..5]
        for r in range(N_BOX):
            for c in range(N_PT):
                if not mask[1, r, c]:
                    pool.append((1, r, c))

        if not pool:
            raise RuntimeError("No available moves (action mask fully blocked).")
        return self.rng.choice(pool)
