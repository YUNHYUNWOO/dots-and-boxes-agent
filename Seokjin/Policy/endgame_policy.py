# endgame_policy.py (single-format: edges only)
from __future__ import annotations
from typing import Dict, List, Tuple, Set, Optional
import numpy as np
import random

# =============================================================================
# Observation helpers (NEW format only)
# observation["edges"]: (n+1, n+1, 2) bool  (z=0: H, z=1: V)
# 내부 계산은 (ori,r,c) [ori=0 H, 1 V]로 통일하고, 반환 시 (c,r,z)로 변환.
# =============================================================================

def _h_v_from_edges(observation: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    E = np.asarray(observation["edges"], dtype=bool)        # (n+1, n+1, 2)
    assert E.ndim == 3 and E.shape[-1] == 2, f"edges shape must be (n+1,n+1,2); got {E.shape}"
    n = E.shape[0] - 1
    h_drawn = E[:, :n, 0]    # (n+1, n)   H(r,c)  r:0..n,   c:0..n-1
    v_drawn = E[:n, :, 1]    # (n, n+1)   V(r,c)  r:0..n-1, c:0..n
    return h_drawn, v_drawn

def _n_from_edges(observation: Dict[str, np.ndarray]) -> int:
    return int(np.asarray(observation["edges"]).shape[0] - 1)

def _ocr_to_crz(edge_ocr: Tuple[int,int,int]) -> Tuple[int,int,int]:
    ori, r, c = edge_ocr
    z = 0 if ori == 0 else 1
    return (c, r, z)

def _mask_allows(mask: np.ndarray, edge_ocr: Tuple[int,int,int]) -> bool:
    # Simulate/Env에서 mask[c,r,z]가 False=가능 / True=불가
    c, r, z = _ocr_to_crz(edge_ocr)
    return (not mask[c, r, z])

# =============================================================================
# Box utilities
# =============================================================================

def _box_edge_counts(h_drawn: np.ndarray, v_drawn: np.ndarray) -> np.ndarray:
    """counts[r,c] = # of drawn edges around box (r,c)."""
    n = h_drawn.shape[1]
    counts = np.zeros((n, n), dtype=int)
    counts += h_drawn[:n, :].astype(int)       # top    H(r,c)
    counts += h_drawn[1:n+1, :].astype(int)    # bottom H(r+1,c)
    counts += v_drawn[:, :n].astype(int)       # left   V(r,c)
    counts += v_drawn[:, 1:n+1].astype(int)    # right  V(r,c+1)
    return counts

# =============================================================================
# Safe / Capture
# =============================================================================

def list_safe_moves(observation: Dict[str, np.ndarray]) -> List[Tuple[int,int,int]]:
    """
    Safe move = 이 수를 두면 인접 박스의 edge-count가 모두 {0,1}.
    (2->3, 3->4 유발 금지)
    반환: (ori,r,c)
    """
    h_drawn, v_drawn = _h_v_from_edges(observation)
    n = h_drawn.shape[1]
    counts = _box_edge_counts(h_drawn, v_drawn)

    safe: List[Tuple[int,int,int]] = []

    # Horizontal
    for r in range(n + 1):
        for c in range(n):
            if not h_drawn[r, c]:
                adj = []
                if 0 <= r - 1 < n: adj.append(counts[r-1, c])  # 위 박스
                if 0 <= r     < n: adj.append(counts[r,   c])  # 아래 박스
                if all(cnt <= 1 for cnt in adj):
                    safe.append((0, r, c))

    # Vertical
    for r in range(n):
        for c in range(n + 1):
            if not v_drawn[r, c]:
                adj = []
                if 0 <= c - 1 < n: adj.append(counts[r, c-1])  # 왼 박스
                if 0 <= c     < n: adj.append(counts[r, c])    # 오른 박스
                if all(cnt <= 1 for cnt in adj):
                    safe.append((1, r, c))

    return safe

def list_capture_moves(observation: Dict[str, np.ndarray]) -> List[Tuple[int,int,int]]:
    """
    Capture move = 이 수를 두면 인접 박스 중 하나 이상이 완성(count 3 -> 4).
    반환: (ori,r,c)
    """
    h_drawn, v_drawn = _h_v_from_edges(observation)
    n = h_drawn.shape[1]
    counts = _box_edge_counts(h_drawn, v_drawn)

    caps: List[Tuple[int,int,int]] = []

    # Horizontal
    for r in range(n + 1):
        for c in range(n):
            if not h_drawn[r, c]:
                adj = []
                if 0 <= r - 1 < n: adj.append(counts[r-1, c])
                if 0 <= r     < n: adj.append(counts[r,   c])
                if any(cnt == 3 for cnt in adj):
                    caps.append((0, r, c))

    # Vertical
    for r in range(n):
        for c in range(n + 1):
            if not v_drawn[r, c]:
                adj = []
                if 0 <= c - 1 < n: adj.append(counts[r, c-1])
                if 0 <= c     < n: adj.append(counts[r, c])
                if any(cnt == 3 for cnt in adj):
                    caps.append((1, r, c))

    return caps

# =============================================================================
# Chain/Loop decomposition
# =============================================================================

def _adjacent_boxes_and_edges(n_box: int, h_drawn: np.ndarray, v_drawn: np.ndarray):
    """
    Undrawn 내부 엣지로 연결된 박스 그래프 인접 리스트와,
    각 박스의 미그은 엣지 리스트를 만든다.
    """
    box_undrawn: Dict[Tuple[int,int], List[Tuple[int,int,int]]] = {}
    for r in range(n_box):
        for c in range(n_box):
            und: List[Tuple[int,int,int]] = []
            if not h_drawn[r, c]:     und.append((0, r,   c))   # H(r,c)
            if not h_drawn[r+1, c]:   und.append((0, r+1, c))   # H(r+1,c)
            if not v_drawn[r, c]:     und.append((1, r,   c))   # V(r,c)
            if not v_drawn[r, c+1]:   und.append((1, r,   c+1)) # V(r,c+1)
            box_undrawn[(r, c)] = und

    adj: Dict[Tuple[int,int], Set[Tuple[int,int]]] = { (r,c): set() for r in range(n_box) for c in range(n_box) }

    def add_internal_edge(e: Tuple[int,int,int]):
        ori, r, c = e
        if ori == 0:
            # H(r,c) 경계 박스: (r-1,c), (r,c)
            if 0 <= r-1 < n_box and 0 <= r < n_box:
                adj[(r-1,c)].add((r,c))
                adj[(r,c)].add((r-1,c))
        else:
            # V(r,c) 경계 박스: (r,c-1), (r,c)
            if 0 <= c-1 < n_box and 0 <= c < n_box:
                adj[(r,c-1)].add((r,c))
                adj[(r,c)].add((r,c-1))

    for r in range(n_box):
        for c in range(n_box):
            for e in box_undrawn[(r,c)]:
                add_internal_edge(e)

    def boxes_of_edge(e: Tuple[int,int,int]) -> List[Tuple[int,int]]:
        ori, r, c = e
        boxes = []
        if ori == 0:
            if 0 <= r-1 < n_box: boxes.append((r-1, c))
            if 0 <= r   < n_box: boxes.append((r,   c))
        else:
            if 0 <= c-1 < n_box: boxes.append((r, c-1))
            if 0 <= c   < n_box: boxes.append((r, c))
        return boxes

    return adj, box_undrawn, boxes_of_edge

def decompose_components(observation: Dict[str, np.ndarray]):
    """
    컴포넌트 리스트 반환. 각 원소는 dict:
      - type: "chain" | "loop"
      - boxes: set[(r,c)]
      - length: 박스 수
      - internal_undrawn_edges: List[(ori,r,c)]
      - open_edges_to_outside:  List[(ori,r,c)]
    """
    h_drawn, v_drawn = _h_v_from_edges(observation)
    n_box = h_drawn.shape[1]
    counts = _box_edge_counts(h_drawn, v_drawn)

    adj, box_undrawn, boxes_of_edge = _adjacent_boxes_and_edges(n_box, h_drawn, v_drawn)

    live = { (r,c) for r in range(n_box) for c in range(n_box) if len(box_undrawn[(r,c)]) > 0 }

    seen: Set[Tuple[int,int]] = set()
    comps: List[Dict] = []

    for start in live:
        if start in seen: continue
        # BFS
        q = [start]
        comp_boxes: Set[Tuple[int,int]] = set()
        seen.add(start)
        while q:
            u = q.pop()
            comp_boxes.add(u)
            for v in adj[u]:
                if v in live and v not in seen:
                    seen.add(v)
                    q.append(v)

        internal_undrawn_edges: List[Tuple[int,int,int]] = []
        open_edges: List[Tuple[int,int,int]] = []

        comp_set = set(comp_boxes)
        for (br, bc) in comp_boxes:
            cand_edges = [
                (0, br,   bc),
                (0, br+1, bc),
                (1, br,   bc),
                (1, br,   bc+1),
            ]
            for e in cand_edges:
                # undrawn only
                ori, r, c = e
                if (ori == 0 and h_drawn[r, c]) or (ori == 1 and v_drawn[r, c]):
                    continue
                boxes = boxes_of_edge(e)
                in_cnt = sum((b in comp_set) for b in boxes)
                if in_cnt == len(boxes) and len(boxes) == 2:
                    internal_undrawn_edges.append(e)
                else:
                    open_edges.append(e)

        # chain vs loop: comp 내부 박스 중 "밖으로 열린 엣지"에 닿는 박스 수로 판단(엔드포인트 유무)
        touched: Set[Tuple[int,int]] = set()
        for e in open_edges:
            for b in boxes_of_edge(e):
                if b in comp_set:
                    touched.add(b)
        comp_type = "chain" if len(touched) > 0 else "loop"

        comps.append({
            "type": comp_type,
            "boxes": comp_set,
            "length": len(comp_set),
            "internal_undrawn_edges": internal_undrawn_edges,
            "open_edges_to_outside": open_edges,
        })

    comps.sort(key=lambda x: (0 if x["type"] == "chain" else 1, x["length"]))
    return comps

# =============================================================================
# Controlled value & opening rule
# =============================================================================

def controlled_value(components: List[Dict]) -> int:
    if not components:
        return 0
    sumv = 0
    has_chain = False
    has_3chain = False
    for comp in components:
        L = comp["length"]
        if comp["type"] == "chain":
            has_chain = True
            if L == 3: has_3chain = True
            sumv += (L - 2)
        else:
            sumv += (L - 4)
    if not has_chain:
        sumv += 8
    elif has_3chain:
        sumv += 6
    else:
        sumv += 4
    return int(sumv)

def choose_endgame_opening(components: List[Dict]) -> int:
    if not components:
        raise ValueError("No components to open.")
    loops = [(i,c) for i,c in enumerate(components) if c["type"] == "loop"]
    chains = [(i,c) for i,c in enumerate(components) if c["type"] == "chain"]
    if loops:
        loops.sort(key=lambda t: t[1]["length"])
        return loops[0][0]
    three_chains = [(i,c) for i,c in chains if c["length"] == 3]
    if three_chains:
        three_chains.sort(key=lambda t: t[1]["length"])
        return three_chains[0][0]
    chains.sort(key=lambda t: t[1]["length"])
    return chains[0][0]

def pick_opening_edge_for_component(observation: Dict[str, np.ndarray], component: Dict) -> Tuple[int,int,int]:
    if component["type"] == "chain":
        if component["open_edges_to_outside"]:
            return component["open_edges_to_outside"][0]
        return component["internal_undrawn_edges"][0]
    return component["internal_undrawn_edges"][0]

# =============================================================================
# Policy
# =============================================================================

class EndgamePolicy:
    """
    0) 3변 칸 있으면 캡처 먼저
    1) 안전수 있으면 그 중 랜덤
    2) 안전수 0 → 체인/루프 분해 후 오프닝 선택
    3) 그래도 없으면 가능한 수 중 랜덤
    """
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def get_action(self, observation: Dict[str, np.ndarray], info: Dict, env) -> Tuple[int,int,int]:
        mask = info["action_mask"]  # shape: (n+1, n+1, 2); False=가능 / True=불가  :contentReference[oaicite:1]{index=1}

        # 0) 캡처 우선
        caps = list_capture_moves(observation)
        if caps:
            cands = [e for e in caps if _mask_allows(mask, e)]
            if cands:
                return _ocr_to_crz(self.rng.choice(cands))

        # 1) 안전수 중 랜덤
        safe = list_safe_moves(observation)
        if safe:
            cands = [e for e in safe if _mask_allows(mask, e)]
            if cands:
                return _ocr_to_crz(self.rng.choice(cands))

        # 2) 엔드게임: 컴포넌트 분해 후 오프닝
        comps = decompose_components(observation)
        if comps:
            idx = choose_endgame_opening(comps)
            move = pick_opening_edge_for_component(observation, comps[idx])  # (ori,r,c)
            if _mask_allows(mask, move):
                return _ocr_to_crz(move)
            # 동일 컴포넌트 내 폴백
            for e in (comps[idx]["open_edges_to_outside"] + comps[idx]["internal_undrawn_edges"]):
                if _mask_allows(mask, e):
                    return _ocr_to_crz(e)

        # 3) 최종 폴백: 전체 스캔
        n = _n_from_edges(observation)
        pool: List[Tuple[int,int,int]] = []
        # H
        for r in range(n+1):
            for c in range(n):
                e = (0, r, c)
                if _mask_allows(mask, e):
                    pool.append(_ocr_to_crz(e))
        # V
        for r in range(n):
            for c in range(n+1):
                e = (1, r, c)
                if _mask_allows(mask, e):
                    pool.append(_ocr_to_crz(e))

        if not pool:
            raise RuntimeError("No available moves found (action mask fully blocked).")
        return self.rng.choice(pool)
