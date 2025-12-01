from typing import Any, Callable, Iterable, Tuple, Optional, NamedTuple, List
from collections import OrderedDict
from dotsandboxes import DotsAndBoxesEngine

EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2
        

class TTEntry:
    __slots__ = ("value", "depth", "flag", "best_action")
    def __init__(self, value: int, depth: int, flag, best_action: Optional[Tuple[int,int,int]]):
        self.value = value          # Evaluated score from the board and min, max flag
        self.depth = depth          # Minimum search depth for which this entry remains valid.
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
        
        # A TT entry is stored only when one of the following conditions is met:
        # - greater search depth than the existing one
        # - a better bound the existing one (LOWER/UPPERBOUND → EXACT)
        replace = False
        if (prev is None):
            replace = True
        else :
            if depth > prev.depth:
                replace = True
            elif depth == prev.depth:
                # EXACT is more valuable
                if flag == EXACT and prev.flag != EXACT:
                    replace = True
                else:
                    replace = False
            else:
                replace = False

        if replace:
            self._t[k] = TTEntry(value, depth, flag, best_action)

        # move entry to the most recently used position
        self._t.move_to_end(k, last=True)
        # if the capacity is exceeded, remove the least recently used entry
        if len(self._t) > self.capacity:
            self._t.popitem(last=False)

    def pv_move(self, eng, maximizing) -> Optional[Tuple[int,int,int]]:
        ent = self._t.get(self.key_from(eng, maximizing))
        return None if ent is None else ent.best_action
