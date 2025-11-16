from DotsAndBoxes import DotsAndBoxesEngine
from typing import Any, Callable, Iterable, Tuple, Optional, NamedTuple, List
from .SearchEngine import BaseSearchEngine
from Util.DnB_Engine_Util import *
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from Util.DnB_Engine_Util import N_BOX
from Policy.Scheduler import Budget_Scheduler

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