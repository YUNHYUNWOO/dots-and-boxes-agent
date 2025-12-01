from typing import Any, Dict

from config import Action, Board, DnBEngineState, N
from dotsandboxes import DotsAndBoxesEngine
from search import BaseSearchEngine
from util.bit_dnb_util import encode_board
from util.scheduler import BaseScheduler
from util.time_manager import TimeManager

from .basepolicy import BasePolicy

# =======================================


class SearchPolicy(BasePolicy):
    """Use a search engine to select actions with a configurable schedule."""

    def __init__(self, search_engine: BaseSearchEngine, config_schedule: dict) -> None:
        super().__init__()
        # Additional dependencies can be attached here if needed.
        self.eng = DotsAndBoxesEngine()
        self.search_engine = search_engine
        self.config_schedule = config_schedule

    def get_config(self, t: int) -> Dict[str, Any]:
        """Resolve dynamic scheduler values for the current ply ``t``."""

        config: Dict[str, Any] = {}
        for k, v in self.config_schedule.items():
            if isinstance(v, BaseScheduler):
                # Resolve scheduler values at the current ply.
                config[k] = v.value(t)
            else:
                # Use static values as-is.
                config[k] = v

        return config

    def get_action(self, observation: Dict[str, Any], time_manager: TimeManager) -> Action:
        """Derive the best move from the search engine for the current observation."""

        # The observation carries the visible board state and auxiliary info,
        # including an action mask for valid moves.
        bitBoard = encode_board(observation['board'])

        state = DnBEngineState(board=bitBoard,
                               cur_player=observation['cur_player'])
        self.eng.set_state(state)

        def get_t(board: Board) -> int:
            t = 0
            for c in range(N):
                for r in range(N):
                    for d in range(2):
                        if board[c][r][d] == 1:
                            t += 1
            return t
        t = get_t(observation['board'])

        config = self.get_config(t)

        self.search_engine.configure(**config)

        best_action, _ = self.search_engine.search(eng=self.eng, time_manager=time_manager)

        return best_action

    def get_log(self):
        """Return and reset the search engine's internal log."""

        log = self.search_engine.get_log()
        self.search_engine.reset_log()
        return log

