"""Alpha-Beta search engine tailored for Dots and Boxes."""

import random
import time
from typing import Callable, List, Optional, Tuple

from config import Action, BitBoard, DnBEngineState
from dotsandboxes import DotsAndBoxesEngine
from heuristic.search_hearistic import complete_extension, default_move_ordering
from util.bit_dnb_util import bit_get_legal_actions, bit_count_edges, bit_count_edges
from util.budget_manager import BaseBudgetManager
from util.time_manager import TimeManager

from .TranspositionTable import EXACT, LOWERBOUND, UPPERBOUND, TranspositionTable
from .search_engine import BaseSearchEngine


class AB_SearchEngine(BaseSearchEngine):
    """Iterative deepening alpha-beta search with optional extensions and TT."""

    def __init__(self) -> None:
        self.tt = TranspositionTable()
        self._tt_reset_keys = ["evaluate"]

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
        self.extension_list: list[Callable] = None
        self.extension_limit: int = 3
        self.use_pvs: bool = False

        self.use_aspiration: bool = False
        self.aspiration_window: float = 1.0

        # Logging variables
        self._nodes = 0
        self._cutoffs = 0
        self._tt_hits = 0
        self._tt_cutoffs = 0
        self._skipped_move = 0
        self._searched_d = 0
        self._time_spent = 0
        self._pvs_nodes = 0
        self._pvs_fallout = 0
        self._aspiration_nodes = 0
        self._aspiration_fall_low = 0
        self._aspiration_fall_high = 0
        self._value = 0

    def search(
        self, eng: DotsAndBoxesEngine, time_manager: TimeManager
    ) -> Tuple[Action, float]:
        """Search for the best action given the current engine state."""

        assert self.evaluate is not None
        assert self.depth is not None

        state = eng.get_state()

        if self.move_ordering is None:
            self.move_ordering = default_move_ordering

        # Budget Assign
        t = bit_count_edges(eng.get_state().board)
        budget = (
            self.budget_manager.get_budget(t, time_manager)
            if self.use_time_control and self.budget_manager
            else float("inf")
        )
        time_manager.set_deadline(budget)

        actions: Optional[List[Action]] = None
        vals: Optional[List[float]] = None

        try:
            if self.use_iterative_deepening:
                last_score = 0
                for d in range(1, self.depth + 1):
                    time_manager.check_time()
                    if self.use_aspiration:
                        actions, vals = self.aspiration_search(eng, d, last_score, time_manager)                      
                    else: 
                        actions, vals = self.alpha_beta(
                            eng=eng,
                            depth=d,
                            root_player=state.cur_player,
                            alpha=-10**9,
                            beta=10**9,
                            time_manager=time_manager,
                            extension_cnt=0,
                        )
                    last_score = vals[0]
                    self._searched_d = d
            else:
                if self.use_aspiration:
                    actions, vals = self.aspiration_search(eng, self.depth, 0, time_manager)
                else:
                    actions, vals = self.alpha_beta(
                        eng=eng,
                        depth=self.depth,
                        root_player=state.cur_player,
                        alpha=-10**9,
                        beta=10**9,
                        time_manager=time_manager,
                        extension_cnt=0,
                    )
        except TimeoutError:
            pass
        eng.set_state(state=state)
        # When there is no time to search any Action.
        if actions is None or vals is None:
            ent = self.tt.probe(eng, True, 0)
            if ent is None or ent.flag != EXACT: actions = bit_get_legal_actions(eng.get_state().board)[:1]
            # actions = bit_get_legal_actions(eng.get_state().board)[0:1]
            else : actions = [ent.best_action]
            vals = [0]

        self._time_spent = time.perf_counter() - time_manager.get_move_start_time()
        self._value = vals[0]
        
        return actions[0], vals[0]
    
    def aspiration_search(self, eng:DotsAndBoxesEngine, depth: int, last_score: float, time_manager: TimeManager):
        self._aspiration_nodes += 1

        alpha = last_score - self.aspiration_window
        beta = last_score + self.aspiration_window
        
        state = eng.get_state()


        actions, vals = self.alpha_beta(
            eng=eng,
            depth=depth,
            root_player=state.cur_player,
            alpha=alpha,
            beta=beta,
            time_manager=time_manager,
            extension_cnt=0,
        )

        if vals[0] <= alpha:
            self._aspiration_fall_low += 1
            actions, vals = self.alpha_beta(
                eng=eng,
                depth=depth,
                root_player=state.cur_player,
                alpha=-10**9,
                beta=beta,
                time_manager=time_manager,
                extension_cnt=0,
            )
        elif vals[0] >= beta:
            self._aspiration_fall_high += 1

            actions, vals = self.alpha_beta(
                eng=eng,
                depth=depth,
                root_player=state.cur_player,
                alpha=alpha,
                beta=10**9,
                time_manager=time_manager,
                extension_cnt=0,
            )
    
        return actions, vals
    def alpha_beta(
        self,
        eng: DotsAndBoxesEngine,
        depth: int,
        root_player: int,
        alpha: float,
        beta: float,
        extension_cnt: int,
        time_manager: TimeManager,
    ) -> Tuple[Optional[list[Action]], list[float]]:
        
        self._nodes += 1
        # check time every 512 nodes.
        if (self._nodes % 512) == 0:
            time_manager.check_time()

        # When meets leafnode
        if depth == 0 or eng.is_game_over():
            sign = 1 if root_player == eng.cur_player else -1
            val = sign * self.evaluate(eng) * self.w_eval
            val += random.random() * 1e-10 if not self.deterministic else 0
            return None, [val]

        maximizing = eng.cur_player == root_player

        # TT(Transposition Table) Search
        ent = self.tt.probe(eng=eng, maximizing=maximizing, depth=depth)
        if ent is not None and ent.depth >= depth:
            self._tt_hits += 1

            if ent.flag == EXACT:
                return [ent.best_action], [ent.value]
            if ent.flag == LOWERBOUND:
                if ent.value >= beta:
                    self._tt_cutoffs += 1
                    return [ent.best_action], [ent.value]
                alpha = max(alpha, ent.value)
            elif ent.flag == UPPERBOUND:
                if ent.value <= alpha:
                    self._tt_cutoffs += 1
                    return [ent.best_action], [ent.value]
                beta = min(beta, ent.value)

        best_vals: List[float] = [
            -10**9 if maximizing else 10**9 for _ in range(self.k)
        ]
        best_actions: List[Action | None] = [None for _ in range(self.k)]


        # Move Ordering
        # After move_ordering is done, pv_action(previous best action) are added to in front of actions list
        # duplication will occur but is negligible
        actions = bit_get_legal_actions(eng.get_state().board)
        move_order = self.move_ordering(actions, eng, self.tt, depth, root_player)
        pv_action = self.tt.pv_move(eng, maximizing)
        if pv_action is not None:
            move_order.insert(0, pv_action)
        # After move ordering is finished, we attach the 'skipped' flag to move_order.
        # The skipped flag is initially set to False.
        move_order = [(False, action) for action in move_order]

        first_move = True
        flag = EXACT
        for skipped, a in move_order:
            player_before = eng.cur_player
            out = eng.apply_action(a)

            # Skip Move
            if self.skip_move:
                n_maximizing = root_player == eng.cur_player
                n_depth = depth - 1
                ent = self.tt.probe(eng=eng, maximizing=n_maximizing, depth=n_depth)
                if ent is not None and ent.depth >= n_depth:
                    # Skip not very promising move
                    if not skipped and (
                        (not maximizing and ent.flag == LOWERBOUND)
                        or (maximizing and ent.flag == UPPERBOUND)
                    ):
                        self._skipped_move += 1
                        move_order.append((True, a))
                        eng.undo_action(a, player_before)
                        continue

            # ab_search's value evaluation is designed as values from this board not cumulated score value.
            # So we need to make Alpha, Beta child to get rid of the score from this board
            immediate_val = len(out["completed_boxes"])
            sign = 1 if root_player == player_before else -1
            alpha_child = alpha - sign * immediate_val
            beta_child = beta - sign * immediate_val

            next_depth = depth - 1
            next_ext_cnt = extension_cnt
            
            # if use_extension is true, depth will not shrink when one of extension functions are true 
            # but there is limit. if extension_cnt >= extension_limit, depth will shrink again
            if self.use_extension and extension_cnt < self.extension_limit:
                for extension in self.extension_list:
                    if extension(out, a):
                        next_depth = depth
                        next_ext_cnt = extension_cnt + 1
                        break

            # if use_pvs is true, first move is searched with the full window
            # and all others searched with the narrow window 
            if not self.use_pvs or first_move:
                _, vals = self.alpha_beta(
                    eng,
                    next_depth,
                    root_player,
                    alpha_child,
                    beta_child,
                    time_manager=time_manager,
                    extension_cnt=next_ext_cnt,
                )
                first_move = False
                val = vals[0]
            else:
                if maximizing:
                    narrow_alpha = alpha_child
                    narrow_beta = alpha_child + 0.000001
                else:
                    narrow_alpha = beta_child - 0.000001
                    narrow_beta = beta_child
                self._pvs_nodes += 1
                _, vals = self.alpha_beta(
                    eng,
                    next_depth,
                    root_player,
                    narrow_alpha,
                    narrow_beta,
                    time_manager=time_manager,
                    extension_cnt=next_ext_cnt,
                )
                val = vals[0]

                val_narrow = sign * immediate_val + val
                need_full = False
                if maximizing:
                    if alpha < val_narrow < beta:
                        need_full = True
                else:
                    if beta > val_narrow > alpha:
                        need_full = True

                # when val_narrow falls out of narrow window,
                # perform full search
                if need_full:
                    self._pvs_fallout += 1
                    _, vals = self.alpha_beta(
                        eng,
                        next_depth,
                        root_player,
                        alpha_child,
                        beta_child,
                        time_manager=time_manager,
                        extension_cnt=next_ext_cnt,
                    )
                    val = vals[0]

            val = sign * immediate_val + val

            # Dnb Engine must be undone.
            # to ensure that its state remains unchanged.
            eng.undo_action(a, player_before)

            # The search engine stores the top-k values for debugging.
            if maximizing:
                self._update_topk(
                    best_actions=best_actions,
                    best_vals=best_vals,
                    a=a,
                    val=val,
                    k=self.k,
                    maximizing=maximizing,
                )
                alpha = max(alpha, best_vals[0])
            else:
                self._update_topk(
                    best_actions=best_actions,
                    best_vals=best_vals,
                    a=a,
                    val=val,
                    k=self.k,
                    maximizing=maximizing,
                )
                beta = min(beta, best_vals[0])

            if alpha >= beta:
                self._cutoffs += 1
                flag = LOWERBOUND if maximizing else UPPERBOUND
                break

        self.tt.store(
            eng,
            maximizing=maximizing,
            depth=depth,
            flag=flag,
            value=best_vals[0],
            best_action=best_actions[0],
        )
        return best_actions, best_vals

    def configure(self, **kwargs):
        """Update configuration and clear TT when necessary."""

        need_reset = False
        for k, v in kwargs.items():
            setattr(self, k, v)
            if k in self._tt_reset_keys:
                need_reset = True
        if need_reset:
            self.clear_tt()
        return self

    def get_log(self):
        """Return search statistics since the last reset."""

        return {
            "nodes": self._nodes,
            "cutoffs": self._cutoffs,
            "tt_hits": self._tt_hits,
            "tt_cutoffs": self._tt_cutoffs,
            "skipped_move": self._skipped_move,
            "depth": self._searched_d,
            "time_spent": self._time_spent,
            "pvs_nodes": self._pvs_nodes,
            "pvs_fallout": self._pvs_fallout,
            "aspiration_nodes": self._aspiration_nodes,
            "aspiration_fall_low": self._aspiration_fall_low,
            "aspiration_fall_high": self._aspiration_fall_high,
            "value": self._value
        }

    def reset_log(self):
        """Reset statistics counters."""

        self._nodes = 0
        self._cutoffs = 0
        self._tt_hits = 0
        self._tt_cutoffs = 0
        self._skipped_move = 0
        self._searched_d = 0
        self._time_spent = 0
        self._pvs_nodes = 0
        self._pvs_fallout = 0
        self._aspiration_nodes = 0
        self._aspiration_fall_low = 0
        self._aspiration_fall_high = 0
        self._value = 0


    def clear_tt(self):
        """Reinitialize the transposition table."""

        self.tt = TranspositionTable()

    def _update_topk(self, best_actions, best_vals, a, val, k, maximizing=True):
        """Maintain the top-k principal variations."""

        for idx, (aa, vv) in enumerate(zip(best_actions, best_vals)):
            if aa == a:
                if maximizing and val <= vv:
                    return
                if not maximizing and val >= vv:
                    return
                best_actions.pop(idx)
                best_vals.pop(idx)
                break

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

        if len(best_vals) > k:
            best_actions.pop()
            best_vals.pop()
