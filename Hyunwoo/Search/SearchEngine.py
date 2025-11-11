from dataclasses import dataclass
from DotsAndBoxes import DotsAndBoxes, DotsAndBoxesEngine, _h_index, _v_index

from typing import Any, Callable, Iterable, Tuple, Optional, NamedTuple, List
from Util import BitFunc


class BaseSearchEngine():
    def __init__(self):
        pass

    def search(self, eng: DotsAndBoxesEngine, state):
        raise NotImplementedError

