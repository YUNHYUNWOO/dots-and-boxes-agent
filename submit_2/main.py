from typing import List, Tuple, Dict, Optional, Any
import random
import numpy as np
import torch
from collections import OrderedDict
import time
from scipy.stats import skewnorm

model = None
TIME_LIMIT = 24.0
time_used = 0.0
time_manager = None

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
    boxes = []
    if d == H:
        if 0 <= r - 1 < N_BOX: boxes.append((c, r - 1))
        if 0 <= r < N_BOX:     boxes.append((c, r))
    else:
        if 0 <= c - 1 < N_BOX: boxes.append((c - 1, r))
        if 0 <= r < N_BOX and 0 <= c < N_BOX: boxes.append((c, r))
    return boxes

def is_box_complete(h_bits: int, v_bits: int, bc: int, br: int) -> bool:

    h1 = (h_bits >> h_index(bc, br)) & 1
    h2 = (h_bits >> h_index(bc, br + 1)) & 1
    v1 = (v_bits >> v_index(bc, br)) & 1
    v2 = (v_bits >> v_index(bc + 1, br)) & 1

    return (h1 & h2 & v1 & v2) == 1

def count_completed_boxes(h_bits: int, v_bits: int) -> int:
    cnt = 0
    for br in range(N_BOX):
        for bc in range(N_BOX):
            if is_box_complete(h_bits, v_bits, bc, br):
                cnt += 1
    return cnt

def edge_is_claimed(edges, c: int, r: int, d: int) -> bool:
    if d == H: return ((edges[0] >> h_index(c, r)) & 1) == 1
    else:      return ((edges[1] >> v_index(c, r)) & 1) == 1

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


## Util/Chain
#######################
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

            if len(open_dirs) == 0:
                # 이미 완성된 박스 -> 체인/루프 후보 아님
                continue

            # 엔드게임 쪽 체인분해는 보통 '1개 또는 2개의 열린 변'만을 다룬다.
            # (3개 이상이면 아직 미들게임 safe 영역)
            # 필요하면 여기서 필터링:

            if len(open_dirs) != 2: continue


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
    visited = { box_id(r,c): False for r in range(N_BOX) for c in range(N_BOX) }
    components = []  # 각 컴포넌트는 [box_id1, box_id2, ...] 리스트

    for r in range(N_BOX):
        for c in range(N_BOX):
            u = box_id(r,c)
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
        if num_deg1 == 2 and len(num_other) == 0:
            res.append({
                'type': 'chain',
                'length': len(comp)
            }) 

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
        
            if (comp['type'] == 'loop'):
                fcv += comp['length'] - 8
        return fcv

    def get_tb(comps):
        tb = 0
        for comp in comps:
            if (comp['type'] == 'chain'):
                tb = 4
        
            if (comp['type'] == 'loop'):
                tb = 8
                break
        return tb
    
    cv = get_fcv(comps) + get_tb(comps)
    return cv
########################################

class DotsAndBoxesEngine:
    def __init__(self, state: Optional[Dict] = None):
        self.h_bits: int = 0
        self.v_bits: int = 0
        self.h_mask = (1 << H_COUNT) - 1  # 30 bits
        self.v_mask = (1 << V_COUNT) - 1  # 30 bits
        self.cur_player: int = 0
        self.score: List[int] = [0, 0]
        if state is not None:
            self.set_state(state)

    # ---- State I/O ----
    def get_state(self) -> Dict:
        return {
            "edges": [self.h_bits, self.v_bits],
            "cur_player": self.cur_player,
            "score": self.score[:],
        }

    def set_state(self, state: Dict) -> None:
        """
        state = {
          "edges": [h, v],       # required
          "cur_player": 0|1,     # required
          "score": [p0, p1],     # required
        }
        검증:
          - h,v는 유효 비트 수 이내여야 함
          - score 합 == 현재 비트보드에서 완성된 박스 수
        """

        # --- edges ---
        h, v = state["edges"]

        h_mask = (1 << H_COUNT) - 1  # 30 bits
        v_mask = (1 << V_COUNT) - 1  # 30 bits

        # 초과 비트가 켜져 있으면 잘못된 

        s0, s1 = state["score"]

        # 일단 반영 후 일관성 검사
        self.h_bits = h
        self.v_bits = v
        self.cur_player = state["cur_player"]
        self.score = [s0, s1]

        # 박스 개수 일관성 체크
        completed = count_completed_boxes(self.h_bits, self.v_bits)

    @classmethod
    def from_state(cls, state: Dict):
        eng = cls()
        eng.set_state(state)
        return eng

    # ---- Internals ----

    def _set_edge(self, c: int, r: int, d: int) -> None:
        if d == H: self.h_bits |= (1 << h_index(c, r))
        else:      self.v_bits |= (1 << v_index(c, r))

    def _clear_edge(self, c: int, r: int, d: int) -> None:
        if d == H: self.h_bits &= ~(1 << h_index(c, r))
        else:      self.v_bits &= ~(1 << v_index(c, r))

    def is_game_over(self) -> bool:
        return (self.h_bits == self.h_mask) and \
           (self.v_bits == self.v_mask)

    # ---- API ----
    def apply_action(self, action: Tuple[int, int, int]) -> Dict:
        c, r, d = action
        check_bounds(c, r, d)

        self._set_edge(c, r, d)

        completed = [] 

        h1 = (self.h_bits >> h_index(0, 0)) & 1
        h2 = (self.h_bits >> h_index(0, 0 + 1)) & 1
        v1 = (self.v_bits >> v_index(0, 0)) & 1
        v2 = (self.v_bits >> v_index(0 + 1, 0)) & 1

        for (bc, br) in boxes_adjacent_to_edge(c, r, d):
            if is_box_complete(self.h_bits, self.v_bits, bc, br):
                completed.append((bc, br))

        if completed:
            self.score[self.cur_player] += len(completed)
            made_box = True
        else:
            made_box = False
            self.cur_player = 1 - self.cur_player

        over = self.is_game_over()

        box_mask = 0
        for (bc, br) in completed:
            box_mask |= (1 << (br * N_BOX + bc))

        return {
            "state": self.get_state(),
            "is_game_over": over,
            "is_box_completed": made_box,
            "completed_boxes": completed,
            "completed_box_mask": box_mask,
        }

    def undo_action(self,
                    action: Tuple[int, int, int],
                    completed_boxes: List[Tuple[int, int]],
                    player_turn: int) -> Dict:
        c, r, d = action
        check_bounds(c, r, d)


        self._clear_edge(c, r, d)

        if completed_boxes:
            self.score[player_turn] -= len(completed_boxes)

        self.cur_player = player_turn
        over = self.is_game_over()

        return {"state": self.get_state(), "is_game_over": over}


class BaseSearchEngine():
    def __init__(self):
        pass

    # ✅ attr 기반 일괄 변경
    def configure(self, **kwargs):
        need_reset = False
        for k, v in kwargs.items():
            setattr(self, k, v)

        return self
    
    def search(self, eng: DotsAndBoxesEngine, state):
        raise NotImplementedError
 

Action = List[int]
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2
        

class TTEntry:
    __slots__ = ("value", "depth", "flag", "best_action")
    def __init__(self, value: int, depth: int, flag, best_action: Optional[Tuple[int,int,int]]):
        self.value = value          # root 기준의 미래마진 값
        self.depth = depth          # 이 값이 유효한 최소 보장 깊이
        self.flag = flag            # EXACT / LOWERBOUND / UPPERBOUND
        self.best_action = best_action


class TranspositionTable:
    def __init__(self):
        self._t = OrderedDict()
        self.capacity = 150000

    @staticmethod
    def key_from(eng: DotsAndBoxesEngine, maximizing: int):
        h, v = eng.h_bits, eng.v_bits
        return (h << 33) | (v << 1) | (1 if maximizing else 0)

    def probe(self, eng, maximizing, depth) -> Optional[TTEntry]:
        k = self.key_from(eng, maximizing)
        ent = self._t.get(k)
        if ent is not None and ent.depth >= depth:
            return ent
        return None

    def store(self, eng, maximizing, depth, flag, value, best_action):
        k = self.key_from(eng, maximizing)
        prev = self._t.get(k)
        # 더 깊은(depth 큰) 결과만 덮어씌우자
        if (prev is None) or (depth >= prev.depth):
            self._t[k] = TTEntry(value, depth, flag, best_action)

        self._t[k] = TTEntry(value, depth, flag, best_action)
        # 최근 사용으로 이동
        self._t.move_to_end(k, last=True)
        # 용량 초과 시 LRU부터 제거
        if len(self._t) > self.capacity:
            self._t.popitem(last=False)

    def pv_move(self, eng, maximizing) -> Optional[Tuple[int,int,int]]:
        ent = self._t.get(self.key_from(eng, maximizing))
        return None if ent is None else ent.best_action


def default_move_ordering(actions, eng, tt, depth, root_player):
    return actions



class AB_TT_Search_TC(BaseSearchEngine):

    def __init__(self):
        self.tt = TranspositionTable()
        self._tt_reset_keys = ['evaluate']
        self.evaluate = None
        self.move_ordering = None
        self.depth = None
        self.use_iterative_deepening = False
        self.deterministic = False
        self.k = 5
        self.T = 0.01
        self.skip_move = True
        self.w_eval = 1
        self.budget_scheduler = Budget_Scheduler(num_turns=60, center=28, scale=7, alpha=1, p=0.3)
        self.use_time_control = True

        ## Loging
        self.nodes = 0
        self.cutoffs = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.skipped_move = 0
        self.searched_d = 0

    def search(self, eng, state, time_manager):
        
        assert self.evaluate != None
        assert self.depth != None

        if self.move_ordering == None:
            self.move_ordering = default_move_ordering

        actions = None

        def count_moves(edges) -> int:
            h_bits, v_bits = edges
            return h_bits.bit_count() + v_bits.bit_count() 
        t = count_moves(eng.get_state()['edges'])
        
        budget = self.get_budget_for_this_move(t, time_manager)
        # print('t', t, 'budget', budget)
        
        self.deadline = time_manager._move_start + budget if self.use_time_control else float('inf')

        print(f't: {t}, remaining: {time_manager.remaining()} budget, {budget}')
        try:
            if self.use_iterative_deepening:
                actions, vals = None, None
                for d in range(self.depth + 1):
                    self._check_time()
                    
                    actions, vals = self.alpha_beta(eng=eng, 
                                    depth=d,
                                    root_player=state['cur_player'],
                                    alpha= -10**9,
                                    beta= 10**9)
                    self.searched_d = d
            else :
                actions, vals = self.alpha_beta(eng=eng, 
                                    depth=self.depth,
                                    root_player=state['cur_player'],
                                    alpha= -10**9,
                                    beta= 10**9
                                    )
            # print('actions: ', actions)
            # print('vals: ', vals)

        except TimeoutError:
            pass

        if actions == None:
            # 어떤 깊이도 끝까지 못 돌린 극단 상황
            actions = get_legal_actions(eng.get_state()['edges'])[0:1]
            vals = [0]

        if self.deterministic == True:
            idx = 0
        else:
            v = torch.tensor(vals[:len(actions)], dtype=torch.float32)
            v = v / self.T
            probs = torch.softmax(v, dim=0).numpy()
            idx = np.random.choice(len(actions), p=probs)

        return actions[idx], vals[idx]
    
    def alpha_beta(self,    
               eng: DotsAndBoxesEngine,
               depth: int,
               root_player: int,
               alpha: int = -10**9,
               beta: int = 10**9
               ) -> Tuple[Optional[Action], int]:
        self.nodes += 1
        if (self.nodes & 1023) == 0:  # 1024의 배수일 때
            self._check_time()


        """
        반환: (가치, 최선의 액션)
        - depth: 남은 탐색 깊이
        - root_player: 최상위 호출 시점의 플레이어(평가 기준)
        - 턴 결정은 eng.cur_player를 기준으로 자동 변환
        """

        # 종료 조건
        if depth == 0 or eng.is_game_over():
            sign = 1 if root_player == eng.cur_player else -1
            return None, [sign * self.evaluate(eng) * self.w_eval]

        # 현재 노드가 '최대화'인지 여부
        maximizing = (eng.cur_player == root_player)
        
        ent = self.tt.probe(eng=eng, maximizing=maximizing, depth=depth)
        if ent != None and ent.depth >= depth:
            self.tt_hits += 1
            if ent.flag == EXACT:
                return [ent.best_action], [ent.value]
            elif ent.flag == LOWERBOUND:
                if ent.value >= beta:
                    self.tt_cutoffs += 1
                    return [ent.best_action], [ent.value]  # 이 노드에서 바로 cutoff
                alpha = max(alpha, ent.value)
            elif ent.flag == UPPERBOUND:
                if ent.value <= alpha:
                    self.tt_cutoffs += 1
                    return [ent.best_action], [ent.value]
                beta = min(beta, ent.value)

        best_vals:List[float] = [-10**9 if maximizing else 10**9 for i in range(self.k)]
        best_actions: List[Action] = [None for i in range(self.k)]

        actions = get_legal_actions(eng.get_state()['edges'])

        pv_action = self.tt.pv_move(eng, maximizing)
        if pv_action != None:
            actions.insert(0, pv_action)

        flag = EXACT
        move_order = self.move_ordering(actions, eng, self.tt, depth, root_player)
        move_order = [(False, action) for action in move_order]

        for skipped, a in move_order:

            # 적용
            player_before = eng.cur_player
            out = eng.apply_action(a)

            ## Skipping Suspicious Move
            if self.skip_move:
                n_maximizing = (root_player == eng.cur_player)
                n_depth = depth - 1
                ent = self.tt.probe(eng=eng, maximizing=n_maximizing, depth=n_depth)
                if ent != None and ent.depth >= n_depth:
                    if not skipped and ((not maximizing and ent.flag == LOWERBOUND) or (maximizing and ent.flag == UPPERBOUND)):
                        self.skipped_move += 1
                        move_order.append((True, a))
                        eng.undo_action(a, out["completed_boxes"], player_before)
                        continue 

            _, vals = self.alpha_beta(eng, depth - 1, root_player, alpha, beta)

            val = vals[0]

            immediate_val = len(out["completed_boxes"])
            sign = 1 if (root_player == player_before) else -1
            val = sign * immediate_val + val

            # 되돌리기
            eng.undo_action(a, out["completed_boxes"], player_before)

            # 갱신
            if maximizing:
                self._update_topk(best_actions=best_actions, best_vals=best_vals, a=a, val=val, k=self.k, maximizing=maximizing)
                alpha = max(alpha, best_vals[0])
            else:
                self._update_topk(best_actions=best_actions, best_vals=best_vals, a=a, val=val, k=self.k, maximizing=maximizing)
                beta = min(beta, best_vals[0])

            if alpha >= beta:
                self.cutoffs += 1
                flag = LOWERBOUND if maximizing else UPPERBOUND
                break  # alpha-beta cut

        self.tt.store(eng, maximizing=maximizing, depth=depth, flag=flag, value=best_vals[0], best_action=best_actions[0])
        return best_actions, best_vals
     
    def configure(self, **kwargs):
        # print(self._tt_reset_keyss)
        need_reset = False
        for k, v in kwargs.items():
            setattr(self, k, v)
            if k in self._tt_reset_keys:
                need_reset = True
        if need_reset:
            self.clear_tt()

        return self
    
    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError()

    def get_budget_for_this_move(self, t, time_manager):
        """
            budget_for_this_move(t):
            -> returns budget for turn t
        """
        rem = time_manager.remaining()
        w = self.budget_scheduler.value(t)

        print(rem, w, rem * w)

        budget = rem * w
        MIN_BUDGET = 0.02   # 최소 20ms
        MAX_BUDGET = rem    # 남은 시간 이상은 쓸 수 없음
        budget = max(MIN_BUDGET, min(budget, MAX_BUDGET))
        SAFETY = 0.05
        budget = max(0.0, budget - SAFETY)
        return budget

    def get_log(self):
        return {
            'nodes': self.nodes,
            'cutoffs': self.cutoffs,
            'tt_hits': self.tt_hits,
            'tt_cutoffs': self.tt_cutoffs,
            'skipped_move': self.skipped_move,
            'depth': self.searched_d
        }
    
    def reset_log(self):
        self.nodes = 0
        self.cutoffs = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.skipped_move = 0
        self.searched_d = 0

    def clear_tt(self):
        self.tt = TranspositionTable()

    def _update_topk(self, best_actions, best_vals, a, val, k, maximizing=True):
        # 2-1) 이미 존재하면, 더 좋을 때만 갱신(교체)하고 위치 재정렬
        for idx, (aa, vv) in enumerate(zip(best_actions, best_vals)):
            if aa == a:
                # 더 나쁜 결과면 무시
                if maximizing and val <= vv: 
                    return
                if (not maximizing) and val >= vv:
                    return
                # 더 좋으면 교체 후 위치 재정렬
                best_actions.pop(idx); best_vals.pop(idx)
                break

        # 2-2) 정렬 위치 찾아 삽입 (수동 구현)
        n = len(best_vals) 
        inserted = False
        for i in range(n):
            if (maximizing and val > best_vals[i]) or ((not maximizing) and val < best_vals[i]):
                best_actions.insert(i, a)
                best_vals.insert(i, val)
                inserted = True
                break
        if not inserted:
            best_actions.append(a)
            best_vals.append(val)

        # 2-3) K 초과 시 꼬리 자르기
        if len(best_vals) > k:
            best_actions.pop()
            best_vals.pop()
class TimeManager():
    def __init__(self):
        self.total_budget = 24.0
        self.used_time = 0.0
        self._move_start = None

    def remaining(self):
        return max(0.0, self.total_budget - self.used_time)
    
    def start_move(self):
        self._move_start = time.perf_counter()
    
    def end_move(self):
        dt = time.perf_counter() - self._move_start
        self.used_time += dt
        return dt
    
    def reset(self):
        self.total_budget = 24.0
        self.used_time = 0.0

class BasePolicy():
    def __init__(self):
        ## 필요한거 있으면 추가
        self.time_manager = TimeManager()
        pass

    def get_action(self, observation, info, env, time_manager:TimeManager):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음
        raise NotImplementedError

    def get_log(self):
        return None


def _adjacent_boxes(c: int, r: int, d: int) -> List[Tuple[int, int]]:
    boxes = []
    if d == H:
        if 0 <= r - 1 < N_BOX: boxes.append((c, r - 1))
        if 0 <= r < N_BOX:     boxes.append((c, r))
    else:
        if 0 <= c - 1 < N_BOX: boxes.append((c - 1, r))
        if 0 <= r < N_BOX and 0 <= c < N_BOX: boxes.append((c, r))
    return boxes

def _box_edge_count(hb: int, vb: int, bc: int, br: int) -> int:
    cnt = 0
    # H(br,bc), H(br+1,bc)
    if (hb >> h_index(bc, br)) & 1:       cnt += 1
    if (hb >> h_index(bc, br + 1)) & 1:   cnt += 1
    # V(br,bc), V(br,bc+1)
    if (vb >> v_index(bc, br)) & 1:       cnt += 1
    if (vb >> v_index(bc + 1, br)) & 1:   cnt += 1
    return cnt

def _makes_third_edge(edges, action) -> bool:
    """액션이 인접 박스 중 '3번째 엣지'를 만들어서 상대에게 4번째를 헌납할 위험인지 체크."""
    c, r, d = action
    h, v = edges
    for (bc, br) in _adjacent_boxes(c, r, d):
        if _box_edge_count(h, v, bc, br) == 2:
            # 지금 두면 3이 됨 (위험수)
            return True
    return False

def evaluate_rel(eng: DotsAndBoxesEngine) -> int:
    moves = get_legal_actions(eng.get_state()['edges'])
    edges = eng.get_state()['edges']

    bad_moves = sum(1 for m in moves if _makes_third_edge(edges, m))
    # bad_moves가 적을수록 좋다
    bad_moves /= 100
    return -bad_moves


def evaluate_cv(eng):
    edges = eng.get_state()['edges']

    adj, external_open, is_candidate = init_box_data(edges)
    comps = get_connected_Components(adj, is_candidate)
    comps = classify_component(comps, adj)
    return get_cv(comps)

def compelete_box(edges, action) -> bool:
    c, r, d = action
    hb, vb = edges
    for (bc, br) in _adjacent_boxes(c, r, d):
        if _box_edge_count(hb, vb, bc, br) == 3:
            # 지금 두면 박스가 완성됨
            return True

# def opens_chain(edges, action) -> bool:
#     adj, external_open, is_candidate = init_box_data(edges)
#     comps = get_connected_Components(adj, is_candidate)

#     for comp in comps:       


## Move_Ordering
def move_ordering(actions, eng: DotsAndBoxesEngine, tt: TranspositionTable, depth:int, root_player:int):
    
    state = eng.get_state()
    edges = state["edges"]
    cur_player = state["cur_player"]
    score = state["score"]

    ranked = []

    forced = []
    safe = []
    for a in actions:
        if compelete_box(edges, a):
            forced.append(a)
        
        if not _makes_third_edge(edges, a):
            safe.append(a)

    rest = actions
    rest = [a for a in rest if a not in forced]
    rest = [a for a in rest if a not in safe]

    ranked = forced + safe + rest

    return ranked


# ===============================================
# Todo Chain analysis
# eng.get_state()
    # {
    #     "edges": [self.h_bits, self.v_bits],
    #     "cur_player": self.cur_player,
    #     "score": self.score[:],
    # }


# =======================================


class SearchPolicy(BasePolicy):
    def __init__(self, SearchEngine:BaseSearchEngine, config_schedule: Dict):
        ## 필요한거 있으면 추가
        self.eng = DotsAndBoxesEngine()
        self.SearchEngine = SearchEngine
        self.config_schedule = config_schedule

    def get_config(self, t):
        config = {}
        for k, v in self.config_schedule.items():
            if isinstance(v, BaseScheduler):
                # BaseScheduler의 하위 클래스면 .value(t)
                config[k] = v.value(t)
            else:
                # 상수면 그대로 사용
                config[k] = v

        return config
    
    def get_action(self, observation, info, env, time_manager:TimeManager):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음
        b_edges = encode_Edges(observation['edges'])

        state = {
            'edges': b_edges,
            'cur_player': observation['cur_player'],
            'score': observation['score']
        }
        t = 60 - np.sum(info['action_mask'] == False)
        config = self.get_config(t)

        self.SearchEngine.configure(**config)
        self.eng.set_state(state)

        best_action, best_val = self.SearchEngine.search(eng=self.eng, state=state, time_manager=time_manager)
        
        return best_action, best_val

    def get_log(self):
        log = self.SearchEngine.get_log()
        self.SearchEngine.reset_log()
        return log

def count_box_sides(board_lines: List[List[List[int]]], bx: int, by: int) -> int:
    """
    박스 (bx, by)의 현재 그려진 변의 개수를 센다.
    board_lines[x][y][0] : (x, y)에서 오른쪽으로 가는 가로선
    board_lines[x][y][1] : (x, y)에서 아래로 가는 세로선
    """
    # 노드 기준 크기
    C = len(board_lines)
    R = len(board_lines[0])

    # 박스 개수는 (C-1) x (R-1) 이라고 가정
    # bx in [0, C-2], by in [0, R-2]
    top    = board_lines[bx][by][0]
    bottom = board_lines[bx][by+1][0]
    left   = board_lines[bx][by][1]
    right  = board_lines[bx+1][by][1]
    return top + bottom + left + right


def boxes_touched_by_edge(x: int, y: int, d: int, W: int, H: int):
    """
    edge (x, y, d)가 인접한 박스(칸)들의 (bx, by) 리스트를 리턴.
    W = 박스 가로 개수, H = 박스 세로 개수
    """
    boxes = []
    if d == 0:  # horizontal
        # 위 박스
        if y > 0 and y - 1 < H:
            boxes.append((x, y - 1))
        # 아래 박스
        if y < H:
            boxes.append((x, y))
    else:  # d == 1 vertical
        # 왼쪽 박스
        if x > 0 and x - 1 < W:
            boxes.append((x - 1, y))
        # 오른쪽 박스
        if x < W:
            boxes.append((x, y))
    return boxes


def get_box_missing_edges(board_lines, bx, by):
    """
    박스 (bx, by)가 3변이 그려져 있다면,
    남은 1개의 변(들)을 (x, y, d) 형태로 리턴.
    (실제로는 항상 1개지만 구현상 리스트로)
    """
    missing = []
    # 위, 아래, 왼, 오
    if board_lines[bx][by][0] == 0:       # top
        missing.append((bx, by, 0))
    if board_lines[bx][by+1][0] == 0:     # bottom
        missing.append((bx, by+1, 0))
    if board_lines[bx][by][1] == 0:       # left
        missing.append((bx, by, 1))
    if board_lines[bx+1][by][1] == 0:     # right
        missing.append((bx+1, by, 1))
    return missing


def dots_and_boxes_policy(edges: List[List[List[int]]]):
    """
    board_lines[c][r][d] 입력으로부터 한 수를 선택해서 (x, y, d)를 리턴.
    
    정책:
    1) 3변이 이미 그려진 박스가 있으면 → 그 박스를 완성하는 선 중 하나를 선택.
    2) 없으면 → 안전수(어떤 박스도 3변이 되지 않는 수) 중에서 랜덤.
    3) 안전수도 없으면 → 남은 아무 수나 랜덤.
    """
    C = len(edges)
    R = len(edges[0])
    D = len(edges[0][0])
    assert D == 2, "이 구현은 d=0(h), d=1(v) 두 방향만 가정함."

    W = C - 1  # 박스 가로 개수
    H = R - 1  # 박스 세로 개수

    available_moves = []

    # 1. 모든 아직 안 그려진 선(엣지) 수집
    for x in range(C):
        for y in range(R):
            for d in range(D):
                # 실제로 존재하는 선만 고려 (범위 밖은 패스)
                if d == 0:
                    # horizontal: x in [0, W-1], y in [0, H]
                    if not (0 <= x < W and 0 <= y <= H):
                        continue
                else:
                    # vertical: x in [0, W], y in [0, H-1]
                    if not (0 <= x <= W and 0 <= y < H):
                        continue

                if edges[x][y][d] == 0:  # 아직 안 그려진 선만
                    available_moves.append((x, y, d))

    if not available_moves:
        return None  # 둘 곳이 없음 (게임 종료 상태)

    # 2. 먼저, 3변 박스를 찾아서 완성시킬 수 있는 수들 찾기
    complete_box_moves = []
    for bx in range(W):
        for by in range(H):
            sides = count_box_sides(edges, bx, by)
            if sides == 3:
                missing = get_box_missing_edges(edges, bx, by)
                # missing 중 실제로 available_moves에 있는 것만 사용
                for mv in missing:
                    if mv in available_moves:
                        complete_box_moves.append(mv)

    if complete_box_moves:
        return random.choice(complete_box_moves)

    # 3. 안전수(safe moves) 찾기:
    #    이 수를 두었을 때, 인접한 박스들 중 어느 것도 3변이 되지 않으면 safe.
    safe_moves = []

    for (x, y, d) in available_moves:
        boxes = boxes_touched_by_edge(x, y, d, W, H)
        unsafe = False
        for (bx, by) in boxes:
            sides_before = count_box_sides(edges, bx, by)
            sides_after = sides_before + 1
            if sides_after == 3:
                unsafe = True
                break
        if not unsafe:
            safe_moves.append((x, y, d))

    if safe_moves:
        return random.choice(safe_moves)

    # 4. 안전수도 없으면 그냥 남은 수 중 랜덤
    return random.choice(available_moves)

class OpeningPolicy(BasePolicy):

    def get_action(self, observation: Dict[str, np.ndarray], info: Dict, env, time_manager:TimeManager) -> Tuple[int,int,int]:
        
        return list(dots_and_boxes_policy(observation['edges'])), None


import math
from typing import List, Tuple, Dict, Optional, Any

class BaseScheduler:
    """모든 스케줄러의 베이스. __call__(t) == value(t)."""
    def value(self, t: float) -> float:
        raise NotImplementedError
    
    def __call__(self, t: float) -> float:
        return self.value(t)
    
    def get_config(self) -> Dict:
        raise NotImplementedError


# ---------------------------
# 단일 구간형 스케줄러들
# ---------------------------

class ConstantScheduler(BaseScheduler):
    def __init__(self, v: float):
        self.v = float(v)
    def value(self, t: float) -> float:
        return self.v
    def get_config(self):
        return {"type": "constant", "v": self.v}


class LinearSchedulerInt(BaseScheduler):
    """t0~t1 구간에서 v0->v1 선형 보간."""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, clip: bool = True):
        assert t1 > t0, "t1 must be > t0"
        self.t0, self.v0, self.t1, self.v1, self.clip = float(t0), float(v0), float(t1), float(v1), clip
        self.dt = self.t1 - self.t0
    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return int(self.v0)
            if t >= self.t1: return int(self.v1)
        x = (t - self.t0) / self.dt
        return int(self.v0 + (self.v1 - self.v0) * x)
    def get_config(self):
        return {"type": "linear", "t0": self.t0, "v0": self.v0, "t1": self.t1, "v1": self.v1, "clip": self.clip}

class LinearScheduler(BaseScheduler):
    """t0~t1 구간에서 v0->v1 선형 보간."""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, clip: bool = True):
        assert t1 > t0, "t1 must be > t0"
        self.t0, self.v0, self.t1, self.v1, self.clip = float(t0), float(v0), float(t1), float(v1), clip
        self.dt = self.t1 - self.t0
    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return self.v0
            if t >= self.t1: return self.v1
        x = (t - self.t0) / self.dt
        return self.v0 + (self.v1 - self.v0) * x
    def get_config(self):
        return {"type": "linear", "t0": self.t0, "v0": self.v0, "t1": self.t1, "v1": self.v1, "clip": self.clip}
    

class ExponentialScheduler(BaseScheduler):
    """t0~t1 구간에서 v0 -> v1 지수 보간 (v>0 권장)."""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, clip: bool = True):
        assert t1 > t0, "t1 must be > t0"
        assert v0 != 0 and v1 != 0, "v0, v1 must be non-zero"
        self.t0, self.v0, self.t1, self.v1, self.clip = float(t0), float(v0), float(t1), float(v1), clip
        self.dt = self.t1 - self.t0
        self.r = self.v1 / self.v0
    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return self.v0
            if t >= self.t1: return self.v1
        x = (t - self.t0) / self.dt
        return self.v0 * (self.r ** x)
    def get_config(self):
        return {"type": "exponential", "t0": self.t0, "v0": self.v0, "t1": self.t1, "v1": self.v1, "clip": self.clip}


class ExponentialSchedulerInt(BaseScheduler):
    """t0~t1 구간에서 v0 -> v1 지수 보간 (v>0 권장)."""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, clip: bool = True):
        assert t1 > t0, "t1 must be > t0"
        assert v0 != 0 and v1 != 0, "v0, v1 must be non-zero"
        self.t0, self.v0, self.t1, self.v1, self.clip = float(t0), float(v0), float(t1), float(v1), clip
        self.dt = self.t1 - self.t0
        self.r = self.v1 / self.v0

    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return int(self.v0)
            if t >= self.t1: return int(self.v1)
        x = (t - self.t0) / self.dt
        return int(self.v0 * (self.r ** x))
    def get_config(self):
        return {"type": "exponential", "t0": self.t0, "v0": self.v0, "t1": self.t1, "v1": self.v1, "clip": self.clip}

class PolynomialScheduler(BaseScheduler):
    """다항 보간: v(t)=v0 + (v1-v0)*((t-t0)/(t1-t0))**power"""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, power: float = 2.0, clip: bool = True):
        assert t1 > t0
        self.t0, self.v0, self.t1, self.v1 = float(t0), float(v0), float(t1), float(v1)
        self.power, self.clip = float(power), clip
        self.dt = self.t1 - self.t0
    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return self.v0
            if t >= self.t1: return self.v1
        x = (t - self.t0) / self.dt
        return self.v0 + (self.v1 - self.v0) * (x ** self.power)
    def get_config(self):
        return {"type": "polynomial", "t0": self.t0, "v0": self.v0, "t1": self.t1, "v1": self.v1, "power": self.power, "clip": self.clip}
    


class CosineScheduler(BaseScheduler):
    """코사인 애닐링: v = v_min + 0.5*(v_max - v_min)*(1 + cos(pi * progress + phase))"""
    def __init__(self, t0: float, t1: float, v_min: float, v_max: float, phase: float = 0.0, clip: bool = True):
        assert t1 > t0
        self.t0, self.t1, self.vmin, self.vmax, self.phase, self.clip = float(t0), float(t1), float(v_min), float(v_max), float(phase), clip
        self.dt = self.t1 - self.t0
    def value(self, t: float) -> float:
        if self.clip:
            if t <= self.t0: return self.vmax
            if t >= self.t1: return self.vmin
        x = (t - self.t0) / self.dt
        return self.vmin + 0.5 * (self.vmax - self.vmin) * (1.0 + math.cos(math.pi * x + self.phase))
    def get_config(self):
        return {"type": "cosine", "t0": self.t0, "t1": self.t1, "v_min": self.vmin, "v_max": self.vmax, "phase": self.phase, "clip": self.clip}


class SigmoidScheduler(BaseScheduler):
    """시그모이드 전이: 중앙 t_mid, 기울기 k."""
    def __init__(self, v_low: float, v_high: float, t_mid: float, k: float = 1.0):
        self.vl, self.vh, self.t_mid, self.k = float(v_low), float(v_high), float(t_mid), float(k)
    def value(self, t: float) -> float:
        s = 1.0 / (1.0 + math.exp(-self.k * (t - self.t_mid)))
        return self.vl + (self.vh - self.vl) * s
    def get_config(self):
        return {"type": "sigmoid", "v_low": self.vl, "v_high": self.vh, "t_mid": self.t_mid, "k": self.k}


class InverseSqrtScheduler(BaseScheduler):
    """v = scale / sqrt(t + offset). 보통 옵티마이저 LR에 사용."""
    def __init__(self, scale: float, offset: float = 1.0):
        assert offset >= 0
        self.scale, self.offset = float(scale), float(offset)
    def value(self, t: float) -> float:
        return self.scale / math.sqrt(max(t + self.offset, 1e-12))
    def get_config(self):
        return {"type": "inverse_sqrt", "scale": self.scale, "offset": self.offset}


# ---------------------------
# 복합/구간/주기형 스케줄러들
# ---------------------------


class Budget_Scheduler(BaseScheduler):
    def __init__(self,
                 num_turns: int,
                 center: float = None,
                 scale: float = None,
                 alpha: float = 3.0,
                 p: float = 0.3):
        self.num_turns = num_turns

        t = np.arange(num_turns)
        g = skewnorm.pdf(t, alpha, loc=center, scale=scale)
        
        w = g / g.sum()
        u = np.ones((num_turns,)) / num_turns
        self.w = p * u + w * (1 - p)

    def value(self, t:int)->float:
        idx = int(t)
        if idx < 0:
            idx = 0
        if idx >= self.num_turns:
            idx = self.num_turns - 1

        tail_sum = float(self.w[idx:].sum())
        if tail_sum <= 0:
            return 1.0  # 남은 weight가 없다면 남은 시간 전부 다 써도 된다고 가정

        return float(self.w[idx] / tail_sum)
    
    def get_config(self):
        return None

class StepScheduler(BaseScheduler):
    """계단형: [(t1, v1), (t2, v2), ...], t < t1 -> v1, t1<=t<t2 -> v2 ... 마지막 이상은 마지막 값."""
    def __init__(self, steps: List[Tuple[float, float]]):
        assert len(steps) > 0
        self.steps = sorted((float(t), float(v)) for t, v in steps)
    def value(self, t: float) -> float:
        for tt, vv in self.steps:
            if t < tt:
                return vv
        return self.steps[-1][1]
    def get_config(self):
        return {"type": "step", "steps": self.steps}


class PiecewiseScheduler(BaseScheduler):
    """
    구간별 서로 다른 스케줄러 연결.
    segments: List of (t_start, t_end, scheduler)
    t < 첫 구간 -> 첫 구간 시작값, t > 마지막 구간 -> 마지막 구간 끝값
    """
    def __init__(self, segments: List[Tuple[float, float, BaseScheduler]]):
        assert len(segments) > 0
        self.segments = [(float(a), float(b), s) for a, b, s in segments]
        for a, b, s in self.segments:
            assert b > a
    def value(self, t: float) -> float:
        # 앞/뒤는 고정
        a0, b0, s0 = self.segments[0]
        an, bn, sn = self.segments[-1]
        if t <= a0:
            return s0.value(a0)
        if t >= bn:
            return sn.value(bn)
        for a, b, s in self.segments:
            if a <= t <= b:
                return s.value(t)
        # 이론상 도달X
        return sn.value(bn)
    def get_config(self):
        return {
            "type": "piecewise",
            "segments": [
                {"t_start": a, "t_end": b, "scheduler": s.get_config()} for a, b, s in self.segments
            ],
        }

class PiecewiseConstantScheduler(BaseScheduler):
    """
    시간 구간을 지정해서, 구간마다 미리 정해둔 value(객체)를 반환하는 스케줄러.

    segments: List[(t_start, t_end, value)]
        - t_start <= t < t_end 인 구간에 대해 value 를 리턴
    default_value:
        - 어느 구간에도 속하지 않을 때 사용할 값 (없으면 에러)
    """
    def __init__(
        self,
        segments: List[Tuple[float, float, Any]],
        default_value: Optional[Any] = None,
    ):
        # 구간 정렬 및 검증
        self.segments = sorted(segments, key=lambda x: x[0])
        self.default_value = default_value

        for (s, e, _) in self.segments:
            if s >= e:
                raise ValueError(f"잘못된 구간: start={s}, end={e}")

    def value(self, t: float) -> Any:
        # t가 속한 구간을 찾아서 해당 value를 리턴
        for (start, end, v) in self.segments:
            if start <= t < end:
                return v

        # 아무 구간에도 안 걸리면 default 사용 (없으면 에러)
        if self.default_value is not None:
            return self.default_value

        raise ValueError(f"t={t} 에 해당하는 구간이 없고, default_value도 없음.")

    def get_config(self) -> Dict:
        return {
            "type": "PiecewiseConstantScheduler",
            "segments": [
                {"start": s, "end": e, "value": v}
                for (s, e, v) in self.segments
            ],
            "default_value": self.default_value,
        }
    

class WarmupHoldDecayScheduler(BaseScheduler):
    """
    Linear warmup -> (optional) hold -> cosine decay
    예: LR warmup 후 코사인 감쇠
    """
    def __init__(self, warmup_end: float, hold_end: float, total_end: float,
                 v_warmup_start: float, v_warmup_end: float,
                 v_final: float):
        assert 0 <= warmup_end <= hold_end <= total_end
        self.warm = LinearScheduler(0.0, v_warmup_start, warmup_end, v_warmup_end, clip=True)
        self.hold_end = hold_end
        self.total_end = total_end
        self.v_hold = v_warmup_end
        self.cos = CosineScheduler(hold_end, total_end, v_min=v_final, v_max=v_warmup_end, clip=True)
    def value(self, t: float) -> float:
        if t <= self.warm.t1:
            return self.warm.value(t)
        if t <= self.hold_end:
            return self.v_hold
        if t <= self.total_end:
            return self.cos.value(t)
        return self.cos.value(self.total_end)
    def get_config(self):
        return {
            "type": "warmup_hold_decay",
            "warmup": self.warm.get_config(),
            "hold_end": self.hold_end,
            "total_end": self.total_end,
            "cosine": self.cos.get_config(),
        }


class CyclicalScheduler(BaseScheduler):
    """
    Triangular (sawtooth) 주기 스케줄.
    period: 주기 길이
    step_ratio: 상승 비율 (0~1), 기본 0.5 -> 절반 상승/절반 하강
    """
    def __init__(self, v_min: float, v_max: float, period: float, step_ratio: float = 0.5, start_t: float = 0.0):
        assert v_max >= v_min
        assert period > 0
        assert 0.0 < step_ratio < 1.0
        self.vmin, self.vmax = float(v_min), float(v_max)
        self.period, self.step_ratio, self.start_t = float(period), float(step_ratio), float(start_t)
    def value(self, t: float) -> float:
        x = (t - self.start_t) % self.period
        up_T = self.period * self.step_ratio
        if x <= up_T:
            # 상승: vmin -> vmax
            return self.vmin + (self.vmax - self.vmin) * (x / up_T)
        else:
            # 하강: vmax -> vmin
            d = (x - up_T) / (self.period - up_T)
            return self.vmax - (self.vmax - self.vmin) * d
    def get_config(self):
        return {"type": "cyclical", "v_min": self.vmin, "v_max": self.vmax, "period": self.period, "step_ratio": self.step_ratio, "start_t": self.start_t}


class CosineRestartScheduler(BaseScheduler):
    """
    SGDR 스타일 코사인 리스타트.
    T0: 첫 주기 길이
    T_mult: 매 주기 길이 배수 (예: 2면 50, 100, 200 ...)
    """
    def __init__(self, v_min: float, v_max: float, T0: int, T_mult: float = 2.0, start_t: int = 0):
        assert T0 > 0 and T_mult >= 1.0
        self.vmin, self.vmax = float(v_min), float(v_max)
        self.T0, self.T_mult, self.start_t = int(T0), float(T_mult), int(start_t)
    def value(self, t: int) -> float:
        # 정수 스텝 기준
        step = max(int(t) - self.start_t, 0)
        Ti = self.T0
        acc = 0
        while step >= Ti:
            step -= Ti
            acc += Ti
            Ti = int(Ti * self.T_mult)
        # 현재 주기에서의 코사인
        if Ti <= 0: Ti = 1
        progress = step / Ti
        return self.vmin + 0.5 * (self.vmax - self.vmin) * (1 + math.cos(math.pi * progress))
    def get_config(self):
        return {"type": "cosine_restart", "v_min": self.vmin, "v_max": self.vmax, "T0": self.T0, "T_mult": self.T_mult, "start_t": self.start_t}


class BooleanScheduler(BaseScheduler):
    """
    주어진 시간 t가 지정된 구간 안에 있으면 True, 아니면 False를 반환하는 스케줄러.
    
    Args:
        true_intervals (List[Tuple[float, float]]): True로 반환할 (시작, 끝) 구간 리스트.
        default (bool): 기본값 (어느 구간에도 속하지 않을 때 반환할 값)
        inclusive (bool): 구간의 양 끝 포함 여부 (기본값 True)
    """
    def __init__(self,
                 true_intervals: List[Tuple[float, float]],
                 default: bool = False,
                 inclusive: bool = True):
        self.true_intervals = true_intervals
        self.default = default
        self.inclusive = inclusive

    def value(self, t: float) -> bool:
        for (start, end) in self.true_intervals:
            if self.inclusive:
                if start <= t <= end:
                    return True
            else:
                if start < t < end:
                    return True
        return self.default

    def get_config(self) -> Dict:
        return {
            "type": "boolean",
            "true_intervals": self.true_intervals,
            "default": self.default,
            "inclusive": self.inclusive
        }

class MixedPolicy(BasePolicy):
    def __init__(self, policy_scheduler: PiecewiseConstantScheduler):
        self.policy_scheduler = policy_scheduler

    def get_policy(self, t):
        # print(self.policy_scheduler.get_config())
        return self.policy_scheduler.value(t)
    
    def get_action(self, observation, info, env, time_manager:TimeManager):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음

        t = 60 - np.sum(info['action_mask'] == False)
        policy = self.get_policy(t)
        
        return policy.get_action(observation, info, env, time_manager)
    
# 반드시 init(),run()함수를 구현해줘야 합니다. 없으면 에러가 발생합니다.
def init():
    # << 체점 시 양쪽 에이전트에 대해서 처음 한 번 실행되는 함수입니다. >>
    # 딥러닝을 통해 게임 에이전트 모델을 training하신 경우에는 모델을 델러오고, 평가 모드로 전환하는 부분을 이곳에 넣으셔야 합니다.
    # 딥러닝을 사용하지 않으셨더라도, Model-based AI로 에이전트를 만드신 분들도 이곳에서 모델/데이터 로딩을 하시면 됩니다.
    global model, time_manager
    
    # 예시1: 학습된 모델 로드
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # model_path = os.path.join(current_dir, "weights.pt") # 학습된 모델 파일 이름을 작성하세요.
    # model = torch.load(model_path, map_location="cpu") 
    # 훈련한 모델을 불러오는 경우 *반드시* 위의 방법으로 상대 경로를 지정하여 불러오시기 바랍니다. 양식을 따르지 않을 경우 채점 서버에서 오류가 발생할 수 있습니다.
    # model.eval() # model을 training이 아닌 evalutation 모드로 전환
    
    # 예시2: 학습된 모델을 사용하지 않는 경우

    policy_part1 = OpeningPolicy()
    config = {
        'evaluate':evaluate_rel,
        'move_ordering':move_ordering,
        'depth': 30,
        'use_iterative_deepening': True,
        'deterministic': True,
        'use_time_control': True
    }
    policy_part2 = SearchPolicy(AB_TT_Search_TC(), config)
    policy_scheduler = PiecewiseConstantScheduler([[15, 60, policy_part2]], default_value=policy_part1)
    model = MixedPolicy(policy_scheduler)

    time_manager = TimeManager()
    # model = OpeningPolicy()

    # 위의 코드는 모델을 사용하지 않는다는 의미입니다. 모델이 필요없는 Rule-based AI를 구현하신 분들은 이렇게 작성하시면 됩니다.


def run(board_lines, xsize, ysize):
    # << 에이전트의 차례가 될 때마다 실행되는 함수입니다. >>
    # 함수의 입력은 위와 같이 현재 board의 현재 상태 (놓인 수들)과 보드의 크기가 제공됩니다.
    # board_lines는 3차원 리스트의 형태로, board_lines[x][y][z]은 해당 자리(x, y, z는 아래 설명 참고)에 수가 놓였는지, 놓이지 않았는지에 대한 값으로 0 또는 1을 가집니다.
    # 이러한 입력 값을 바탕으로, 다음과 같이 놓을 수를 반환해주시면 됩니다.
    global time_manager


    def get_init_action_mask(n_box) -> np.ndarray:
        # shape: (n_box+1, n_box+1)
        r = np.arange(n_box + 1)[None, :]        # (R,1)
        c = np.arange(n_box + 1)[:, None]        # (1,C)

        mask = np.zeros((n_box + 1, n_box + 1, 2), dtype=bool)
        # Horizontal mask: True only when c == n_box
        
        mask[:,:,0] = (c == n_box)
        # Vertical mask: True only when r == n_box
        mask[:,:,1] = (r == n_box)
        # Stack so that mask[0] = H, mask[1] = V
        return mask
    
    action_mask = get_init_action_mask(5)

    count = 0
    for r in range(xsize):
        for c in range(ysize):
            for d in range(2):
                if board_lines[r][c][d]:
                    count += 1
                    action_mask[r,c,d] = board_lines[r][c][d]

    if count == 0 or count == 1:
        time_manager.reset()

    time_manager.start_move()
    obs = {
        'edges': board_lines,
        'cur_player': 0,
        'score' : [0, 0]
    }
    info = {
        'action_mask': action_mask
    }
    
    action, _ = model.get_action(observation=obs, info=info, env=None, time_manager=time_manager)

    time_manager.end_move()

    return action


if __name__ == "__main__":
    init()
    for step in range(100000):
        print(step / (100000))
        board_lines = [[[0 for _ in range(2)] for __ in range(6)]  for ___ in range(6)]
        
        for c in range(6):
            for r in range(6):
                for z in range(2):
                    board_lines[c][r][z] = random.randint(0, 1)
                    
        # for z in range(2):
        #     for r in range(6):
        #         for c in range(6):
        #             if z == 0 and c != 5:
        #                 print(board_lines[c][r][z], end=' ')
        #             elif z == 1 and r != 5:
        #                 print(board_lines[c][r][z], end=' ')
        #         print()
                
        # print("Board Lines:")
        # print(board_lines)

        action = run(board_lines, 5, 5)
        #print(action)
        #print(type(action))
        # print(type(action))
        if board_lines[action[0]][action[1]][action[2]]:
            #print("Selected Action:")
            #print(action)
            break  
        c, r, d = action
        if (c == 6 and d == 0) or (r == 6 and d == 1):
            #print('action Out of Bound')
            break

