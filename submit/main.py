from typing import List, Tuple, Dict, Optional, Any
import random
import numpy as np
import torch
from collections import OrderedDict
import time
N_BOX = 5
N = N_BOX + 1
H_COUNT = N * (N - 1)   # 30
V_COUNT = (N - 1) * N   # 30
TOTAL_BOXES = N_BOX * N_BOX  # 25
H, V = 0, 1

model = None
TIME_LIMIT = 24.0
time_used = 0.0

def h_index(c: int, r: int) -> int:
    return r * (N - 1) + c

def v_index(c: int, r: int) -> int:
    return r * N + c

def check_bounds(c: int, r: int, d: int) -> None:
    return None

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
        if not isinstance(state, dict):
            raise TypeError("state must be a dict")

        # --- edges ---
        if "edges" not in state or not isinstance(state["edges"], (list, tuple)) or len(state["edges"]) != 2:
            raise ValueError("state['edges'] must be [h, v]")

        h, v = state["edges"]
        if not isinstance(h, int) or not isinstance(v, int):
            raise TypeError("edges must be integers")


        # 초과 비트가 켜져 있으면 잘못된 상태
        if h & ~self.h_mask:
            raise ValueError("H edges contain out-of-range bits")
        if v & ~self.v_mask:
            raise ValueError("V edges contain out-of-range bits")

        # --- cur_player ---
        if "cur_player" not in state or state["cur_player"] not in (0, 1):
            raise ValueError("state['cur_player'] must be 0 or 1")

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
    def _edge_is_claimed(self, c: int, r: int, d: int) -> bool:
        if d == H: return ((self.h_bits >> h_index(c, r)) & 1) == 1
        else:      return ((self.v_bits >> v_index(c, r)) & 1) == 1

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
        if self._edge_is_claimed(c, r, d):
            raise ValueError("Edge already claimed")

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

        if not self._edge_is_claimed(c, r, d):
            raise ValueError("Cannot undo: Edge is not set")

        self._clear_edge(c, r, d)

        if completed_boxes:
            self.score[player_turn] -= len(completed_boxes)
            if self.score[player_turn] < 0:
                raise ValueError("Undo would make score negative (invalid history)")

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

class TTEntry:
    __slots__ = ("value", "depth", "best_action")
    def __init__(self, value: int, depth: int, best_action: Optional[Tuple[int,int,int]]):
        self.value = value          # root 기준의 미래마진 값
        self.depth = depth          # 이 값이 유효한 최소 보장 깊이
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

    def store(self, eng, maximizing, depth, value, best_action):
        k = self.key_from(eng, maximizing)
        prev = self._t.get(k)
        # 더 깊은(depth 큰) 결과만 덮어씌우자
        if (prev is None) or (depth >= prev.depth):
            self._t[k] = TTEntry(value, depth, best_action)

        self._t[k] = TTEntry(value, depth, best_action)
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
    
class AB_TT_Search(BaseSearchEngine):

    def __init__(self):
        self.tt = TranspositionTable()
        self._tt_reset_keys = ['evaluate']
        self.evaluate = None
        self.move_ordering = lambda x: x
        self.depth = None
        self.use_iterative_deepening = False
        self.deterministic = False
        self.k = 5
        self.T = 0.01

    def search(self, eng, state):


        if self.move_ordering == None:
            self.move_ordering = default_move_ordering

        actions = None
        if self.use_iterative_deepening:
            actions, vals = self.iterative_deepening(eng, state)
        else :
            actions, vals = self.alpha_beta(eng=eng, 
                                depth=self.depth,
                                root_player=state['cur_player'],
                                alpha= -10**9,
                                beta= 10**9
                                )
        # print('actions: ', actions)
        # print('vals: ', vals)

        if self.deterministic == True:
            idx = 0
        else:
            v = torch.tensor(vals[:len(actions)], dtype=torch.float32)
            v = v / self.T
            probs = torch.softmax(v, dim=0).numpy()
            idx = np.random.choice(len(actions), p=probs)

        return actions[idx], vals[idx]

    def iterative_deepening(self, eng, state):
        best_action = None
        for d in range(self.depth + 1):
            best_actions, best_vals = self.alpha_beta(eng=eng, 
                               depth=d,
                               root_player=state['cur_player'],
                               alpha= -10**9,
                               beta= 10**9)
            
        return best_actions, best_vals
    
    def alpha_beta(self,    
               eng: DotsAndBoxesEngine,
               depth: int,
               root_player: int,
               alpha: int = -10**9,
               beta: int = 10**9
               ) -> Tuple[Optional[Action], int]:
        
        """
        반환: (가치, 최선의 액션)
        - depth: 남은 탐색 깊이
        - root_player: 최상위 호출 시점의 플레이어(평가 기준)
        - 턴 결정은 eng.cur_player를 기준으로 자동 변환
        """

        # 종료 조건
        if depth == 0 or eng.is_game_over():
            sign = 1 if root_player == eng.cur_player else -1
            return None, [sign * self.evaluate(eng)]

        # 현재 노드가 '최대화'인지 여부
        maximizing = (eng.cur_player == root_player)
        
        ent = self.tt.probe(eng=eng, maximizing=maximizing, depth=depth)
        if ent != None:
            return [ent.best_action], [ent.value]

        best_vals:List[float] = [-10**9 if maximizing else 10**9 for i in range(self.k)]
        best_actions: List[Action] = [None for i in range(self.k)]

        actions = get_legal_actions(eng.get_state()['edges'])
        
        # print(actions)
        # edges = decode_Edges(eng.get_state()['edges'])

        # for a in actions:
        #     if edges[a[0]][a[1]][a[2]]:
        #         print('Same_Edges_Detected')
        #         print(a)
        #         print('decoded_edges')
        #         for d in range(2):
        #             for c in range(len(edges)):
        #                 for r in range(len(edges)):
        #                     print(edges[c][r][d], end=" ")
        #                 print()
        #             print("=======")
        #         exit()

        pv_action = self.tt.pv_move(eng, maximizing)
        if pv_action != None:
            actions.insert(0, pv_action)

        for a in self.move_ordering(actions, eng, self.tt, depth, root_player):

            # 적용
            player_before = eng.cur_player
            out = eng.apply_action(a)

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

            if beta <= alpha:
                break  # alpha-beta cut
        
        self.tt.store(eng, maximizing=maximizing, depth=depth, value=best_vals[0], best_action=best_actions[0])
        
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


class BasePolicy():
    def __init__(self):
        ## 필요한거 있으면 추가
        pass

    def get_action(self, observation, info, env):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음
        raise NotImplementedError

def _adjacent_boxes(r: int, c: int, d: int) -> List[Tuple[int, int]]:
    boxes = []
    if d == H:
        if 0 <= r - 1 < N_BOX: boxes.append((r - 1, c))
        if 0 <= r < N_BOX:     boxes.append((r, c))
    else:
        if 0 <= c - 1 < N_BOX: boxes.append((r, c - 1))
        if 0 <= r < N_BOX and 0 <= c < N_BOX: boxes.append((r, c))
    return boxes

def _box_edge_count(hb: int, vb: int, br: int, bc: int) -> int:
    cnt = 0
    # H(br,bc), H(br+1,bc)
    if (hb >> h_index(br, bc)) & 1:       cnt += 1
    if (hb >> h_index(br + 1, bc)) & 1:   cnt += 1
    # V(br,bc), V(br,bc+1)
    if (vb >> v_index(br, bc)) & 1:       cnt += 1
    if (vb >> v_index(br, bc + 1)) & 1:   cnt += 1
    return cnt


def _makes_third_edge(eng: DotsAndBoxesEngine, a) -> bool:
    """액션이 인접 박스 중 '3번째 엣지'를 만들어서 상대에게 4번째를 헌납할 위험인지 체크."""
    c, r, d = a
    hb, vb = eng.h_bits, eng.v_bits
    for (br, bc) in _adjacent_boxes(r, c, d):
        if _box_edge_count(hb, vb, br, bc) == 2:
            # 지금 두면 3이 됨 (위험수)
            return True
    return False

def evaluate(eng: DotsAndBoxesEngine, root_player: int) -> int:
    """아주 단순한 평가식: 점수차 크게 + '위험수(3-edge 만드는 수)'는 페널티."""
    me, opp = root_player, 1 - root_player
    score_term = (eng.score[me] - eng.score[opp]) * 100

    moves = get_legal_actions(eng.get_state()['edges'])
    bad_moves = sum(1 for m in moves if _makes_third_edge(eng, m))
    # bad_moves가 적을수록 좋다

    return score_term - bad_moves

def evaluate_rel(eng: DotsAndBoxesEngine) -> int:
    moves = get_legal_actions(eng.get_state()['edges'])
    bad_moves = sum(1 for m in moves if _makes_third_edge(eng, m))
    # bad_moves가 적을수록 좋다
    bad_moves /= 100
    return -bad_moves

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
    
    def get_action(self, observation, info, env):
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

        best_action, _ = self.SearchEngine.search(eng=self.eng, state=state)
        
        return best_action

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

    def get_action(self, observation: Dict[str, np.ndarray], info: Dict, env) -> Tuple[int,int,int]:
        
        return list(dots_and_boxes_policy(observation['edges']))

class BaseScheduler:
    """모든 스케줄러의 베이스. __call__(t) == value(t)."""
    def value(self, t: float) -> float:
        raise NotImplementedError
    
    def __call__(self, t: float) -> float:
        return self.value(t)
    
    def get_config(self) -> Dict:
        raise NotImplementedError

class ExponentialSchedulerInt(BaseScheduler):
    """t0~t1 구간에서 v0 -> v1 지수 보간 (v>0 권장)."""
    def __init__(self, t0: float, v0: float, t1: float, v1: float, clip: bool = True):

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


    def value(self, t: float) -> Any:
        # t가 속한 구간을 찾아서 해당 value를 리턴
        for (start, end, v) in self.segments:
            if start <= t < end:
                return v

        # 아무 구간에도 안 걸리면 default 사용 (없으면 에러)
        if self.default_value is not None:
            return self.default_value

    def get_config(self) -> Dict:
        return {
            "type": "PiecewiseConstantScheduler",
            "segments": [
                {"start": s, "end": e, "value": v}
                for (s, e, v) in self.segments
            ],
            "default_value": self.default_value,
        }
    

class MixedPolicy(BasePolicy):
    def __init__(self, policy_scheduler: PiecewiseConstantScheduler):
        self.policy_scheduler = policy_scheduler

    def get_policy(self, t):
        # print(self.policy_scheduler.get_config())
        return self.policy_scheduler.value(t)
    
    def get_action(self, observation, info, env):
        # observation에는 에이전트가 관측하는 상태 정보
        # info는 그 외에 부가적인 정보들
            # 필수적으로 action mask가 포함되어있음

        t = 60 - np.sum(info['action_mask'] == False)
        policy = self.get_policy(t)
        
        return policy.get_action(observation, info, env)
    
# 반드시 init(),run()함수를 구현해줘야 합니다. 없으면 에러가 발생합니다.
def init():
    # << 체점 시 양쪽 에이전트에 대해서 처음 한 번 실행되는 함수입니다. >>
    # 딥러닝을 통해 게임 에이전트 모델을 training하신 경우에는 모델을 델러오고, 평가 모드로 전환하는 부분을 이곳에 넣으셔야 합니다.
    # 딥러닝을 사용하지 않으셨더라도, Model-based AI로 에이전트를 만드신 분들도 이곳에서 모델/데이터 로딩을 하시면 됩니다.
    global model
    
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
        'move_ordering':None,
        'depth': ExponentialSchedulerInt(15, 2, 45, 10),
        'use_iterative_deepening': True,
        'deterministic': True
    }
    policy_part2 = SearchPolicy(AB_TT_Search(), config)
    policy_scheduler = PiecewiseConstantScheduler([[20, 60, policy_part2]], default_value=policy_part1)
    model = MixedPolicy(policy_scheduler)

    # model = OpeningPolicy()

    # 위의 코드는 모델을 사용하지 않는다는 의미입니다. 모델이 필요없는 Rule-based AI를 구현하신 분들은 이렇게 작성하시면 됩니다.


def run(board_lines, xsize, ysize):
    # << 에이전트의 차례가 될 때마다 실행되는 함수입니다. >>
    # 함수의 입력은 위와 같이 현재 board의 현재 상태 (놓인 수들)과 보드의 크기가 제공됩니다.
    # board_lines는 3차원 리스트의 형태로, board_lines[x][y][z]은 해당 자리(x, y, z는 아래 설명 참고)에 수가 놓였는지, 놓이지 않았는지에 대한 값으로 0 또는 1을 가집니다.
    # 이러한 입력 값을 바탕으로, 다음과 같이 놓을 수를 반환해주시면 됩니다.
    global time_used

    start = time.perf_counter()

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
        time_used = 0.0

    obs = {
        'edges': board_lines,
        'cur_player': 0,
        'score' : [0, 0]
    }
    info = {
        'action_mask': action_mask
    }

    remaining = TIME_LIMIT - time_used
    if remaining <= 0.1:
        return list(dots_and_boxes_policy(obs['edges']))
    
    action = model.get_action(observation=obs, info=info, env=None)
    end = time.perf_counter()
    time_used += (end - start)
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

