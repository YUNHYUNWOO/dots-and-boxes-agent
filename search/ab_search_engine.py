from typing import Any, Callable, Iterable, Tuple, Optional, NamedTuple, List
import random

import numpy as np
import torch
import torch.nn as nn

from config import N_BOX
from util.bit_dnb_util import *
from util.time_manager import TimeManager
from dotsandboxes import DotsAndBoxesEngine
from util.budget_manager import BaseBudgetManager
from heuristic.search_hearistic import give_away_extension, complete_extension, default_move_ordering

from .search_engine import BaseSearchEngine
from .TranspositionTable import TranspositionTable, EXACT, LOWERBOUND, UPPERBOUND



class AB_SearchEngine(BaseSearchEngine):

    def __init__(self):
        self.tt = TranspositionTable()
        self._tt_reset_keys = ['evaluate']

        # configuration 
        self.evaluate: Callable[[DotsAndBoxesEngine], float] | None = None
        self.move_ordering: Callable[
            [list[Action], DotsAndBoxesEngine, TranspositionTable, int, int],
            list[Action],
        ] | None = None

        self.depth: int | None = None
        self.use_iterative_deepening: bool = False
        self.deterministic: bool = False

        self.k: int = 5
        self.skip_move: bool = False
        self.w_eval: float = 1.0

        self.use_time_control: bool = False
        self.budget_manager: BaseBudgetManager | None = None

        self.use_extension: bool = False
        self.extension_limit: int = 5
        self.use_pvs: bool = False

        ## Logging variables
        self._nodes = 0
        self._cutoffs = 0
        self._tt_hits = 0
        self._tt_cutoffs = 0
        self._skipped_move = 0
        self._searched_d = 0

    def search(self, eng, state, time_manager: TimeManager):
        
        assert self.evaluate != None
        assert self.depth != None

        if self.move_ordering == None:
            self.move_ordering = default_move_ordering

        actions = None

        def count_moves(edges) -> int:
            h_bits, v_bits = edges
            return h_bits.bit_count() + v_bits.bit_count() 
    
        t = count_moves(eng.get_state().board)

        budget = self.budget_manager.get_budget(t, time_manager)
        time_manager.set_deadline(budget if self.use_time_control else float('inf'))
        
        try:
            if self.use_iterative_deepening:
                actions, vals = None, None
                    
                for d in range(1, self.depth + 1):
                    time_manager.check_time()

                    actions, vals = self.alpha_beta(eng=eng, 
                            depth=d,
                            root_player=state.cur_player,
                            alpha= -10**9,
                            beta= 10**9,
                            time_manager=time_manager,
                            extension_cnt=0)
                    
                    self._searched_d = d
            else :
                actions, vals = self.alpha_beta(eng=eng, 
                                    depth=self.depth,
                                    root_player=state.cur_player,
                                    alpha= -10**9,
                                    beta= 10**9,
                                    time_manager=time_manager,
                                    extension_cnt=0)
                                    

        except TimeoutError:
            pass

        # When time is Not enough to search any move
        if actions == None:
            actions = bit_get_legal_actions(eng.get_state().board)[0:1]
            vals = [0]
        
        return actions[0], vals[0]
    
    def alpha_beta(self,    
               eng: DotsAndBoxesEngine,
               depth: int,
               root_player: int,
               alpha: int,
               beta: int,
               extension_cnt,
               time_manager:TimeManager
               ) -> Tuple[Optional[list[Action]], list[float]]:
        self._nodes += 1
        if (self._nodes % 512) == 0:  # 1024의 배수일 때
            time_manager.check_time()

        """
        반환: (가치, 최선의 액션)
        - depth: 남은 탐색 깊이
        - root_player: 최상위 호출 시점의 플레이어(평가 기준)
        - 턴 결정은 eng.cur_player를 기준으로 자동 변환
        """

        # 종료 조건
        if depth == 0 or eng.is_game_over():
            sign = 1 if root_player == eng.cur_player else -1
            val = sign * self.evaluate(eng) * self.w_eval
            val += random.random() * 1e-10 if not self.deterministic else 0

            return None, [val]

        # 현재 노드가 '최대화'인지 여부
        maximizing = (eng.cur_player == root_player)
        
        ent = self.tt.probe(eng=eng, maximizing=maximizing, depth=depth)
        if ent != None and ent.depth >= depth:
            self._tt_hits += 1

            if ent.flag == EXACT:
                return [ent.best_action], [ent.value]
            
            elif ent.flag == LOWERBOUND:
                if ent.value >= beta:
                    self._tt_cutoffs += 1
                    return [ent.best_action], [ent.value]  # 이 노드에서 바로 cutoff
                alpha = max(alpha, ent.value)

            elif ent.flag == UPPERBOUND:
                if ent.value <= alpha:
                    self._tt_cutoffs += 1
                    return [ent.best_action], [ent.value]
                beta = min(beta, ent.value)

        best_vals: List[float] = [-10**9 if maximizing else 10**9 for i in range(self.k)]
        best_actions: List[Action] = [None for i in range(self.k)]

        actions = bit_get_legal_actions(eng.get_state().board)
        # random.shuffle(actions)

        move_order = self.move_ordering(actions, eng, self.tt, depth, root_player)
        pv_action = self.tt.pv_move(eng, maximizing)
        if pv_action != None:
            move_order.insert(0, pv_action)
        move_order = [(False, action) for action in move_order]
        
        first_move = True
        flag = EXACT
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
            
            immediate_val = len(out["completed_boxes"])
            sign = 1 if (root_player == player_before) else -1
            alpha_child = alpha - sign * immediate_val
            beta_child  = beta  - sign * immediate_val    

            next_depth = depth - 1
            next_ext_cnt = extension_cnt
            if self.use_extension and extension_cnt < self.extension_limit and \
                depth == 1 and \
               (complete_extension(out, a)):
                next_depth = depth      # 같은 depth로 재탐색
                next_ext_cnt = extension_cnt + 1

            if not self.use_pvs or first_move:
                _, vals = self.alpha_beta(eng, next_depth, root_player, alpha_child, beta_child, time_manager=time_manager, extension_cnt=next_ext_cnt)
                first_move = False
                val = vals[0]
            else:
                if maximizing:
                    narrow_alpha = alpha_child
                    narrow_beta = alpha_child + 1
                else:
                    narrow_alpha = beta_child - 1
                    narrow_beta  = beta_child

                # (a) zero-window 검색
                _, vals = self.alpha_beta(
                    eng, next_depth, root_player,
                    narrow_alpha, narrow_beta,
                    time_manager=time_manager, 
                    extension_cnt=next_ext_cnt
                )
                val = vals[0]

                val_narrow = sign * immediate_val + val
                need_full = False
                if maximizing:
                    # max 노드: 현재 best(alpha)를 넘어섰고, beta보다 작으면 정확한 값 필요
                    if val_narrow > alpha and val_narrow < beta:
                        need_full = True
                else:
                    # min 노드: 현재 best(beta)보다 낮았고, alpha보다 크면 정확한 값 필요
                    if val_narrow < beta and val_narrow > alpha:
                        need_full = True

                if need_full:
                    _, vals = self.alpha_beta(
                        eng, next_depth, root_player,
                        alpha_child, beta_child,
                        time_manager=time_manager, 
                        extension_cnt=next_ext_cnt
                    )
                    val = vals[0]

            val = sign * immediate_val + val

            # 되돌리기
            eng.undo_action(a, player_before)

            # 갱신
            if maximizing:
                self._update_topk(best_actions=best_actions, best_vals=best_vals, a=a, val=val, k=self.k, maximizing=maximizing)
                alpha = max(alpha, best_vals[0])
            else:
                self._update_topk(best_actions=best_actions, best_vals=best_vals, a=a, val=val, k=self.k, maximizing=maximizing)
                beta = min(beta, best_vals[0])

            if alpha >= beta:
                self._cutoffs += 1
                flag = LOWERBOUND if maximizing else UPPERBOUND
                break  # alpha-beta cut

        self.tt.store(eng, maximizing=maximizing, depth=depth, flag=flag, value=best_vals[0], best_action=best_actions[0])
        return best_actions, best_vals
    
    def configure(self, **kwargs):
        need_reset = False
        for k, v in kwargs.items():
            setattr(self, k, v)
            if k in self._tt_reset_keys:
                need_reset = True
        if need_reset:
            self.clear_tt()

        return self

    def get_log(self):
        return {
            'nodes': self._nodes,
            'cutoffs': self._cutoffs,
            'tt_hits': self._tt_hits,
            'tt_cutoffs': self._tt_cutoffs,
            'skipped_move': self._skipped_move,
            'depth': self._searched_d
        }
    
    def reset_log(self):
        self._nodes = 0
        self._cutoffs = 0
        self._tt_hits = 0
        self._tt_cutoffs = 0
        self._skipped_move = 0
        self._searched_d = 0

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