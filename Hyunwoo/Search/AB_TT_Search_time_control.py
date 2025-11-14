from DotsAndBoxes import DotsAndBoxesEngine
from typing import Any, Callable, Iterable, Tuple, Optional, NamedTuple, List
from .SearchEngine import BaseSearchEngine
from Util.DnB_Engine_Util import *
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

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
        
        assert self.evaluate != None
        assert self.depth != None

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