from DotsAndBoxes import DotsAndBoxesEngine
from typing import Any, Callable, Iterable, Tuple, Optional, NamedTuple, List
from .SearchEngine import BaseSearchEngine
from Util.DnB_Engine_Util import *

Action = List[int]

class AlphaBetaSearch(BaseSearchEngine):

    def __init__(self, evaluate, move_ordering, depth):
        self.evaluate = evaluate
        self.move_ordering = move_ordering
        if move_ordering == None:
            self.move_ordering = lambda x: x
        self.depth = depth

    def search(self, eng, state):
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
            return None, self.evaluate(eng, root_player)

        # 현재 노드가 '최대화'인지 여부
        maximizing = (eng.cur_player == root_player)

        best_val = -10**9 if maximizing else 10**9
        best_action: Optional[Action] = None

        actions = get_legal_actions_encoded(eng.get_state()['edges'])
        
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

            # 되돌리기
            eng.undo_action(a, out["completed_boxes"], player_before)

            # 갱신
            if maximizing:
                if val > best_val:
                    best_action, best_val,  = a, val
                alpha = max(alpha, best_val)
            else:
                if val < best_val:
                    best_action, best_val = a, val
                beta = min(beta, best_val)

            if beta <= alpha:
                break  # alpha-beta cut

        return best_action, best_val