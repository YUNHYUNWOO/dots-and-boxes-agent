# policy_v1.py — ChainPolicy (6x6 dots / 5x5 boxes)
from __future__ import annotations
from typing import Dict, Tuple, Optional, List, Set, Dict as TDict
import numpy as np
import random

Action = Tuple[int, int, int]   # (c, r, z) where z: 0=H, 1=V
Box = Tuple[int, int]           # (r, c)


class ChainPolicy:
    """
    Priority:
      0) Capture FIRST (count==3 박스 완성)
         - Safe moves가 남아있으면 캡처 1개만 하고 종료 (pre-endgame).
         - Safe moves가 없으면(=엔드게임) 컨트롤러 플랜(더블크로스: 체인 2, 루프 4 남김) 실행/인계.
      1) Safe move (두어도 어떤 박스도 3변이 되지 않음)
      2) Endgame opener:
         - UNDRAWN 엣지 그래프(OUTSIDE 포함)로 체인/루프 분해.
         - CV = sum(L-2)+sum(ℓ-4) - TB, TB=4(체인≥1)/8(루프만)/0(없음)
         - 오프너 결정:
             * CV ≥ 2: 최단 체인(없으면 최단 루프)
             * CV ≤ 1: 3-체인이 있으면 3-체인, 없으면 최단 루프
             * 특수형: 체인 1개 + 루프≥1 & CV ≤ 1 → 루프부터
         - 오프닝 엣지:
             * 체인: 경계로 열린 미그어진 엣지(open_edges_to_outside) 우선 → 없으면 내부 undrawn
             * 루프: 내부 undrawn
         - 가능한 한 비캡처(non-capturing) 엣지를 우선 시도.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.controller_plan: Optional[TDict] = None
        self.last_structure: Optional[TDict] = None

    # ---------- Public API ----------
    def get_action(self, observation: Dict, info: Dict, env) -> Action:
        mask: np.ndarray = info["action_mask"]
        h_drawn, v_drawn = self._h_v_from_edges(observation)
        counts = self._box_edge_counts(h_drawn, v_drawn)

        # 6x6 dots → 5x5 boxes
        assert counts.shape == (5, 5), f"Expected 5x5 boxes, got {counts.shape}"

        # 0) 캡처 최우선
        capture_any = self._find_any_capture(counts, h_drawn, v_drawn, mask)
        safe_moves = self._collect_safe_moves(h_drawn, v_drawn, counts, mask)


        if capture_any is not None:
            if safe_moves:
                return capture_any
            # 엔드게임: 컨트롤러 플랜
            if self.controller_plan is None:
                self.controller_plan = self._plan_from_capture_seed(capture_any, counts, h_drawn, v_drawn)
            move = self._controller_capture_step(counts, h_drawn, v_drawn, mask)
            if move is not None:
                return move
            move = self._controller_handover_step(counts, h_drawn, v_drawn, mask)
            if move is not None:
                self.controller_plan = None
                return move
            move = self._pick_any_noncapturing(mask, counts, h_drawn, v_drawn)
            if move is not None:
                self.controller_plan = None
                return move
            self.controller_plan = None
            return self._pick_random_legal(mask)

        # 1) 안전수
        if safe_moves:
            self.controller_plan = None
            return self.rng.choice(safe_moves)

        # 2) 엔드게임: 분해 + CV + 오프너 선택 + 오프닝 엣지
        struct = self._decompose_components_undrawn(h_drawn, v_drawn)
        self.last_structure = struct
        comps = struct["components"]
        if not comps:
            return self._pick_random_legal(mask)

        cv = self._controlled_value_from_components(comps)
        comp_choice = self._choose_component_to_open_legacy_rules(comps, cv)
        if comp_choice is None:
            return self._pick_random_legal(mask)

        move = self._choose_opening_edge_boundary_first(comp_choice, counts, h_drawn, v_drawn, mask)
        if move is not None:
            self.controller_plan = None
            return move

        # 같은 컴포넌트 내 다른 엣지들 폴백(마스크/비캡처 제약 충족 위해)
        for e in (comp_choice["open_edges_to_outside"] + comp_choice["internal_undrawn_edges"]):
            if self._mask_allows(mask, (e[1], e[0], e[2])):  # (c,r,z) 변환 주의
                # e는 (ori, r, c), mask는 (c, r, z)
                return (e[2], e[1], 0 if e[0] == 0 else 1)

        noncap = self._pick_any_noncapturing(mask, counts, h_drawn, v_drawn)
        if noncap is not None:
            self.controller_plan = None
            return noncap

        
        return self._pick_random_legal(mask)

    # ---------- Safe & Capture ----------
    def _collect_safe_moves(
        self,
        h_drawn: np.ndarray,
        v_drawn: np.ndarray,
        counts: np.ndarray,
        mask: np.ndarray,
    ) -> List[Action]:
        """두어도 어떤 박스도 3변이 되지 않는 엣지."""
        safe: List[Action] = []
        n = counts.shape[0]

        # Horizontal
        for r in range(n + 1):
            for c in range(n):
                if h_drawn[r, c]:
                    continue
                adj = []
                if r - 1 >= 0:
                    adj.append(counts[r - 1, c])
                if r < n:
                    adj.append(counts[r, c])
                if all(cnt <= 1 for cnt in adj):
                    a = (c, r, 0)
                    if self._mask_allows(mask, a):
                        safe.append(a)

        # Vertical
        for r in range(n):
            for c in range(n + 1):
                if v_drawn[r, c]:
                    continue
                adj = []
                if c - 1 >= 0:
                    adj.append(counts[r, c - 1])
                if c < n:
                    adj.append(counts[r, c])
                if all(cnt <= 1 for cnt in adj):
                    a = (c, r, 1)
                    if self._mask_allows(mask, a):
                        safe.append(a)
        return safe

    def _find_any_capture(
        self,
        counts: np.ndarray,
        h_drawn: np.ndarray,
        v_drawn: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[Action]:
        """count==3 박스를 완성하는 임의의 캡처 한 수."""
        n = counts.shape[0]
        for r in range(n):
            for c in range(n):
                if counts[r, c] != 3:
                    continue
                if not h_drawn[r, c]:
                    a = (c, r, 0)
                    if self._mask_allows(mask, a):
                        return a
                if not h_drawn[r + 1, c]:
                    a = (c, r + 1, 0)
                    if self._mask_allows(mask, a):
                        return a
                if not v_drawn[r, c]:
                    a = (c, r, 1)
                    if self._mask_allows(mask, a):
                        return a
                if not v_drawn[r, c + 1]:
                    a = (c + 1, r, 1)
                    if self._mask_allows(mask, a):
                        return a
        return None

    # ---------- Endgame decomposition (UNDRAWN graph with OUTSIDE) ----------
    def _decompose_components_undrawn(
        self,
        h_drawn: np.ndarray,
        v_drawn: np.ndarray,
    ) -> TDict:
        """
        예전 방식:
          - 노드: 모든 박스 (r,c)
          - 내부 미그어진 변으로 이웃 박스 연결
          - 경계 미그어진 변은 OUTSIDE(None)과 연결
          - 컴포넌트 판별:
              * OUTSIDE와 연결되면 chain
              * OUTSIDE 연결이 없고 내부 차수가 모두 2면 loop (아니면 chain fallback)
        각 컴포넌트:
          type, boxes(set), length(int),
          open_edges_to_outside: [(ori,r,c)], internal_undrawn_edges: [(ori,r,c)]
        """
        n = h_drawn.shape[1]
        adj: TDict[Optional[Box], Set[Optional[Box]]] = {}
        edge_between: TDict[Tuple[Optional[Box], Optional[Box]], List[Tuple[int, int, int]]] = {}

        def add_edge(a: Optional[Box], b: Optional[Box], edge: Tuple[int, int, int]):
            if a not in adj:
                adj[a] = set()
            if b not in adj:
                adj[b] = set()
            adj[a].add(b)
            adj[b].add(a)
            key = (a, b)
            if a is not None and b is not None and b < a:
                key = (b, a)
            edge_between.setdefault(key, []).append(edge)

        # init nodes
        for r in range(n):
            for c in range(n):
                adj[(r, c)] = set()

        # internal shared edges (undrawn)
        for r in range(n):
            for c in range(n):
                # right neighbor: shared vertical (r, c+1)
                if c + 1 < n and not v_drawn[r, c + 1]:
                    add_edge((r, c), (r, c + 1), (1, r, c + 1))
                # bottom neighbor: shared horizontal (r+1, c)
                if r + 1 < n and not h_drawn[r + 1, c]:
                    add_edge((r, c), (r + 1, c), (0, r + 1, c))

        # boundary undrawn edges → OUTSIDE(None)
        for r in range(n):
            for c in range(n):
                if not h_drawn[r, c]:
                    add_edge((r, c), None, (0, r, c))
                if not h_drawn[r + 1, c]:
                    add_edge((r, c), None, (0, r + 1, c))
                if not v_drawn[r, c]:
                    add_edge((r, c), None, (1, r, c))
                if not v_drawn[r, c + 1]:
                    add_edge((r, c), None, (1, r, c + 1))

        visited: Set[Box] = set()
        components: List[TDict] = []

        # candidates: degree>0 (i.e., touching any undrawn edge)
        candidates = [b for b in adj.keys() if b is not None and len(adj[b]) > 0]

        for start in candidates:
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            comp_boxes: Set[Box] = {start}
            touches_outside = False
            degrees_inside: TDict[Box, int] = {}
            inside_edges: Set[Tuple[int, int, int]] = set()
            outside_edges: Set[Tuple[int, int, int]] = set()

            while stack:
                u = stack.pop()
                for v in adj[u]:
                    key = (u, v)
                    if u is not None and v is not None and v < u:
                        key = (v, u)
                    edges_uv = edge_between.get(key, [])
                    if v is None:
                        touches_outside = True
                        for e in edges_uv:
                            outside_edges.add(e)
                    else:
                        for e in edges_uv:
                            inside_edges.add(e)
                        degrees_inside[u] = degrees_inside.get(u, 0) + 1
                        degrees_inside[v] = degrees_inside.get(v, 0) + 1
                        if v not in visited:
                            visited.add(v)
                            comp_boxes.add(v)
                            stack.append(v)

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
                "open_edges_to_outside": list(outside_edges),
                "internal_undrawn_edges": list(inside_edges),
            })

        # 안정적인 순서: 길이 asc, 루프 우선
        components.sort(key=lambda x: (x["length"], 0 if x["type"] == "loop" else 1))
        return {"n": n, "components": components}

    # ---------- CV & Opener choice ----------
    def _controlled_value_from_components(self, components: List[TDict]) -> int:
        sum_ch = sum(c["length"] - 2 for c in components if c["type"] == "chain")
        sum_lp = sum(c["length"] - 4 for c in components if c["type"] == "loop")
        if any(c["type"] == "chain" for c in components):
            tb = 4
        elif any(c["type"] == "loop" for c in components):
            tb = 8
        else:
            tb = 0
        return int(sum_ch + sum_lp - tb)

    def _choose_component_to_open_legacy_rules(self, components: List[TDict], cv: int) -> Optional[TDict]:
        chains = [c for c in components if c["type"] == "chain"]
        loops = [c for c in components if c["type"] == "loop"]
        shortest_chain = min(chains, key=lambda c: c["length"]) if chains else None
        shortest_loop = min(loops, key=lambda c: c["length"]) if loops else None
        c3 = [c for c in chains if c["length"] == 3]

        if cv >= 2:
            return shortest_chain if shortest_chain is not None else shortest_loop

        # cv <= 1
        if len(chains) == 1 and len(loops) >= 1:
            return shortest_loop if shortest_loop is not None else chains[0]
        if c3:
            return c3[0]
        return shortest_loop if shortest_loop is not None else shortest_chain

    def _choose_opening_edge_boundary_first(
        self,
        comp: TDict,
        counts: np.ndarray,
        h_drawn: np.ndarray,
        v_drawn: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[Action]:
        """
        체인: 경계로 열린 미그어진 엣지(open_edges_to_outside) 우선 → 없으면 내부 undrawn
        루프: 내부 undrawn
        가능한 한 비캡처(non-capturing) 엣지 우선, 마스크 허용 필수.
        컴포넌트 엣지는 (ori,r,c); 마스크는 (c,r,z) → 변환 주의.
        """
        def edge_to_action(e: Tuple[int, int, int]) -> Action:
            ori, r, c = e
            return (c, r, 0 if ori == 0 else 1)

        def is_noncapturing(e: Tuple[int, int, int]) -> bool:
            a = edge_to_action(e)
            return self._edge_not_capturing(a, counts, counts.shape[0])

        # 후보 목록 구성: 체인은 outside 먼저
        if comp["type"] == "chain":
            primary = comp["open_edges_to_outside"]
            secondary = comp["internal_undrawn_edges"]
        else:  # loop
            primary = comp["internal_undrawn_edges"]
            secondary = []

        # 1) primary에서 (마스크 허용 & 비캡처) 찾기
        for e in primary:
            a = edge_to_action(e)
            if self._mask_allows(mask, a) and is_noncapturing(e):
                return a
        # 2) primary에서 (마스크 허용) 아무거나
        for e in primary:
            a = edge_to_action(e)
            if self._mask_allows(mask, a):
                return a
        # 3) secondary에서도 같은 순서
        for e in secondary:
            a = edge_to_action(e)
            if self._mask_allows(mask, a) and is_noncapturing(e):
                return a
        for e in secondary:
            a = edge_to_action(e)
            if self._mask_allows(mask, a):
                return a
        return None

    # ---------- Controller (double-cross) ----------
    def _plan_from_capture_seed(
        self,
        capture_move: Action,
        counts: np.ndarray,
        h_drawn: np.ndarray,
        v_drawn: np.ndarray,
    ) -> TDict:
        """
        캡처 직후 열린 성분(2·3변 상자들)을 찾아 leave-2/leave-4 계획을 세움.
        """
        n = counts.shape[0]
        c, r, z = capture_move

        seed_boxes: List[Box] = []
        if z == 0:
            if r - 1 >= 0 and counts[r - 1, c] == 3:
                seed_boxes.append((r - 1, c))
            if r < n and counts[r, c] == 3:
                seed_boxes.append((r, c))
        else:
            if c - 1 >= 0 and counts[r, c - 1] == 3:
                seed_boxes.append((r, c - 1))
            if c < n and counts[r, c] == 3:
                seed_boxes.append((r, c))

        def neighbors_23(b: Box) -> List[Box]:
            br, bc = b
            neigh: List[Box] = []
            if not h_drawn[br, bc] and br - 1 >= 0 and counts[br - 1, bc] in (2, 3):
                neigh.append((br - 1, bc))
            if not h_drawn[br + 1, bc] and br + 1 < n and counts[br + 1, bc] in (2, 3):
                neigh.append((br + 1, bc))
            if not v_drawn[br, bc] and bc - 1 >= 0 and counts[br, bc - 1] in (2, 3):
                neigh.append((br, bc - 1))
            if not v_drawn[br, bc + 1] and bc + 1 < n and counts[br, bc + 1] in (2, 3):
                neigh.append((br, bc + 1))
            return neigh

        visited: Set[Box] = set()
        stack: List[Box] = seed_boxes[:1] if seed_boxes else []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in neighbors_23(u):
                if v not in visited:
                    stack.append(v)

        if not visited:
            visited = set(seed_boxes)
        boxes = visited
        length = len(boxes)

        # classify loop/chain within these boxes
        def deg_and_border(b: Box) -> Tuple[int, int]:
            br, bc = b
            deg = 0
            border = 0
            if not h_drawn[br, bc]:
                if br - 1 >= 0 and (br - 1, bc) in boxes:
                    deg += 1
                else:
                    border += 1
            if not h_drawn[br + 1, bc]:
                if br + 1 < n and (br + 1, bc) in boxes:
                    deg += 1
                else:
                    border += 1
            if not v_drawn[br, bc]:
                if bc - 1 >= 0 and (br, bc - 1) in boxes:
                    deg += 1
                else:
                    border += 1
            if not v_drawn[br, bc + 1]:
                if bc + 1 < n and (br, bc + 1) in boxes:
                    deg += 1
                else:
                    border += 1
            return deg, border

        is_loop = True
        for b in boxes:
            deg, border = deg_and_border(b)
            if not (deg == 2 and border == 0):
                is_loop = False
                break

        leave_k = 4 if is_loop else 2
        return {"type": "loop" if is_loop else "chain", "boxes": set(boxes), "length": length, "leave_k": leave_k}

    def _controller_capture_step(
        self,
        counts: np.ndarray,
        h_drawn: np.ndarray,
        v_drawn: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[Action]:
        """해당 성분에서 leave_k를 남기고 나머지 캡처."""
        if not self.controller_plan:
            return None
        comp = self.controller_plan
        boxes: Set[Box] = comp["boxes"]
        length = comp["length"]
        leave_k = comp["leave_k"]

        captured_so_far = sum(1 for (r, c) in boxes if counts[r, c] == 4)
        need_to_take = max(0, length - leave_k - captured_so_far)
        if need_to_take <= 0:
            return None

        n = counts.shape[0]
        for r, c in boxes:
            if counts[r, c] != 3:
                continue
            if not h_drawn[r, c]:
                a = (c, r, 0)
                if self._mask_allows(mask, a):
                    return a
            if not h_drawn[r + 1, c]:
                a = (c, r + 1, 0)
                if self._mask_allows(mask, a):
                    return a
            if not v_drawn[r, c]:
                a = (c, r, 1)
                if self._mask_allows(mask, a):
                    return a
            if not v_drawn[r, c + 1]:
                a = (c + 1, r, 1)
                if self._mask_allows(mask, a):
                    return a

        return self._find_any_capture(counts, h_drawn, v_drawn, mask)

    def _controller_handover_step(
        self,
        counts: np.ndarray,
        h_drawn: np.ndarray,
        v_drawn: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[Action]:
        """leave_k를 남긴 뒤, 같은 성분에서 비캡처 엣지로 턴을 넘김."""
        if not self.controller_plan:
            return None
        comp = self.controller_plan
        boxes: Set[Box] = comp["boxes"]
        n = counts.shape[0]

        for r, c in boxes:
            if counts[r, c] == 4:
                continue
            # 이 박스의 비캡처 엣지 시도
            if not h_drawn[r, c]:
                a = (c, r, 0)
                if self._mask_allows(mask, a) and self._edge_not_capturing(a, counts, n):
                    return a
            if not h_drawn[r + 1, c]:
                a = (c, r + 1, 0)
                if self._mask_allows(mask, a) and self._edge_not_capturing(a, counts, n):
                    return a
            if not v_drawn[r, c]:
                a = (c, r, 1)
                if self._mask_allows(mask, a) and self._edge_not_capturing(a, counts, n):
                    return a
            if not v_drawn[r, c + 1]:
                a = (c + 1, r, 1)
                if self._mask_allows(mask, a) and self._edge_not_capturing(a, counts, n):
                    return a
        return None

    # ---------- Utilities ----------
    def _mask_allows(self, mask: np.ndarray, action: Action) -> bool:
        c, r, z = action
        return not bool(mask[c, r, z])

    def _edge_not_capturing(self, action: Action, counts: np.ndarray, n: int) -> bool:
        """엣지를 두어도 인접 박스 중 count==3이 없어야 True."""
        c, r, z = action
        if z == 0:  # H
            adj = []
            if r - 1 >= 0:
                adj.append((r - 1, c))
            if r < n:
                adj.append((r, c))
        else:       # V
            adj = []
            if c - 1 >= 0:
                adj.append((r, c - 1))
            if c < n:
                adj.append((r, c))
        for (br, bc) in adj:
            if counts[br, bc] == 3:
                return False
        return True

    def _pick_any_noncapturing(
        self,
        mask: np.ndarray,
        counts: np.ndarray,
        h_drawn: np.ndarray,
        v_drawn: np.ndarray,
    ) -> Optional[Action]:
        """보편 비캡처 폴백."""
        n = counts.shape[0]
        # Horizontal
        for r in range(n + 1):
            for c in range(n):
                if not h_drawn[r, c]:
                    a = (c, r, 0)
                    if self._mask_allows(mask, a) and self._edge_not_capturing(a, counts, n):
                        return a
        # Vertical
        for r in range(n):
            for c in range(n + 1):
                if not v_drawn[r, c]:
                    a = (c, r, 1)
                    if self._mask_allows(mask, a) and self._edge_not_capturing(a, counts, n):
                        return a
        return None

    def _pick_random_legal(self, mask: np.ndarray) -> Action:
        legal: List[Action] = []
        C, R, Z = mask.shape
        for c in range(C):
            for r in range(R):
                for z in range(Z):
                    if not mask[c, r, z]:
                        legal.append((c, r, z))
        if not legal:
            raise RuntimeError("No legal moves in action_mask.")
        return self.rng.choice(legal)

    # ---------- Observation ----------
    def _h_v_from_edges(self, observation: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        observation['edges'] is (col, row, z) from env → transpose to (row, col, z).
        Return: h_drawn (n+1, n), v_drawn (n, n+1)
        """
        E = np.asarray(observation["edges"], dtype=bool)
        assert E.ndim == 3 and E.shape[-1] == 2, f"edges must be (n+1,n+1,2); got {E.shape}"
        Et = E.transpose(1, 0, 2)           # (row, col, z)
        n = Et.shape[0] - 1
        h_drawn = Et[:, :n, 0]               # H(r,c)
        v_drawn = Et[:n, :, 1]               # V(r,c)
        return h_drawn, v_drawn

    def _box_edge_counts(self, h_drawn: np.ndarray, v_drawn: np.ndarray) -> np.ndarray:
        n = h_drawn.shape[1]
        counts = np.zeros((n, n), dtype=np.int8)
        counts += h_drawn[0:n, 0:n].astype(np.int8)
        counts += h_drawn[1:n + 1, 0:n].astype(np.int8)
        counts += v_drawn[0:n, 0:n].astype(np.int8)
        counts += v_drawn[0:n, 1:n + 1].astype(np.int8)
        return counts
