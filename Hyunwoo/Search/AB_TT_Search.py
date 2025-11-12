from DotsAndBoxes import DotsAndBoxesEngine
from typing import Any, Callable, Iterable, Tuple, Optional, NamedTuple, List
from .SearchEngine import BaseSearchEngine
from Util.DnB_Engine_Util import *


Action = List[int]

class TTEntry:
    __slots__ = ("value", "depth", "best_action")
    def __init__(self, value: int, depth: int, best_action: Optional[Tuple[int,int,int]]):
        self.value = value          # root 기준의 미래마진 값
        self.depth = depth          # 이 값이 유효한 최소 보장 깊이
        self.best_action = best_action

class TranspositionTable:
    def __init__(self):
        self._t = {}

    @staticmethod
    def key_from(eng: DotsAndBoxesEngine, maximizing: int):
        h, v = eng.h_bits, eng.v_bits
        return (h, v, maximizing)

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

    def pv_move(self, eng, maximizing) -> Optional[Tuple[int,int,int]]:
        ent = self._t.get(self.key_from(eng, maximizing))
        return None if ent is None else ent.best_action
    


class AB_TT_Search(BaseSearchEngine):

    def __init__(self):
        self.tt = TranspositionTable()
        self._tt_reset_keys = ['evaluate']
        self.evaluate = None
        self.move_ordering = lambda x: x
        self.depth = None



    def search(self, eng, state):

        assert self.evaluate != None
        assert self.depth != None

        if self.move_ordering == None:
            self.move_ordering = lambda x: x

        return self.alpha_beta(eng=eng, 
                               depth=self.depth,
                               root_player=state['cur_player'],
                               alpha= -10**9,
                               beta= 10**9
                               )

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
            return None, sign * self.evaluate(eng)


        # 현재 노드가 '최대화'인지 여부
        maximizing = (eng.cur_player == root_player)

        
        ent = self.tt.probe(eng=eng, maximizing=maximizing, depth=depth)
        if ent != None:
            return ent.best_action, ent.value

        best_val = -10**9 if maximizing else 10**9
        best_action: Optional[Action] = None

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


        for a in self.move_ordering(actions):

            # 적용
            player_before = eng.cur_player
            out = eng.apply_action(a)

            _, val = self.alpha_beta(eng, depth - 1, root_player, alpha, beta)
            immediate_val = len(out["completed_boxes"])
            sign = 1 if (root_player == player_before) else -1
            val = sign * immediate_val + val

            # 되돌리기
            eng.undo_action(a, out["completed_boxes"], player_before)

            # 갱신
            if maximizing:
                if val > best_val:
                    best_action, best_val = a, val
                alpha = max(alpha, best_val)
            else:
                if val < best_val:
                    best_action, best_val = a, val
                beta = min(beta, best_val)

            if beta <= alpha:
                break  # alpha-beta cut
            
        self.tt.store(eng, maximizing=maximizing, depth=depth, value=best_val, best_action=best_action)
        
        return best_action, best_val
     
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