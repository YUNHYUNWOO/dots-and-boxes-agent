import random

from config import Action, Board, N, N_BOX
from util.dnb_util import count_box_edges, get_boxes_adjacent_to_edge, get_missing_edges
from util.time_manager import TimeManager

from .basepolicy import BasePolicy


def dots_and_boxes_policy(board: Board) -> Action:
    """Select a move for the opening phase using simple heuristics.

    The policy inspects ``board[c][r][d]`` to choose an action ``(x, y, d)``.
    Strategy:
    1) If any box already has three edges, draw one of the remaining edges to complete it.
    2) Otherwise, choose randomly among "safe" moves that do not create a three-edged box.
    3) If no safe moves exist, pick any remaining edge uniformly at random.
    """

    available_moves = []

    # 1. Collect every edge that has not been drawn.
    for c in range(N):
        for r in range(N):
            for d in range(2):
                # Consider only edges that exist within board bounds.
                if d == 0:
                    # horizontal: x in [0, W-1], y in [0, H]
                    if not (0 <= c < N_BOX and 0 <= r <= N_BOX):
                        continue
                else:
                    # vertical: x in [0, W], y in [0, H-1]
                    if not (0 <= c <= N_BOX and 0 <= r < N_BOX):
                        continue

                if board[c][r][d] == 0:
                    available_moves.append((c, r, d))

    if not available_moves:
        return None  # No moves available (game finished).

    # 2. Prefer completing boxes that already have three edges.
    complete_box_moves = []
    for br in range(N_BOX):
        for bc in range(N_BOX):
            box = (br, bc)
            sides = count_box_edges(board, box)
            if sides == 3:
                missing = get_missing_edges(board, box)
                for a in missing:
                    if a in available_moves:
                        complete_box_moves.append(a)

    if complete_box_moves:
        return random.choice(complete_box_moves)

    # 3. Find safe moves that do not leave any adjacent box with three edges.
    safe_moves = []

    for action in available_moves:
        boxes = get_boxes_adjacent_to_edge(action)
        unsafe = False
        for box in boxes:
            sides_before = count_box_edges(board, box)
            sides_after = sides_before + 1
            if sides_after == 3:
                unsafe = True
                break
        if not unsafe:
            safe_moves.append(action)

    if safe_moves:
        return random.choice(safe_moves)

    # 4. If no safe move exists, pick any remaining move at random.
    return random.choice(available_moves)


class OpeningPolicy(BasePolicy):
    def get_action(self, observation: dict, time_manager: TimeManager) -> Action:
        """Choose an opening move based on adjacency heuristics."""

        return dots_and_boxes_policy(observation['board'])
