"""Search extensions and move ordering heuristics for Dots and Boxes."""

from config import Action, BitBoard
from dotsandboxes import DotsAndBoxesEngine
from search.TranspositionTable import TranspositionTable
from util.bit_dnb_util import bit_count_box_edges, bit_makes_third_edge
from util.dnb_util import get_boxes_adjacent_to_edge


# Search extensions
def complete_extension(out: dict, action: Action) -> bool:
    """Extend depth if the move completed a box."""

    return bool(out.get("is_box_completed"))


def give_away_extension(out: dict, action: Action) -> bool:
    """Extend depth if the move hands over a nearly complete box."""

    board: BitBoard = out["state"]["board"]
    for box in get_boxes_adjacent_to_edge(action):
        if bit_count_box_edges(board, box) == 3:
            return True
    return False


# Move ordering
def default_move_ordering(
    actions: list[Action],
    eng: DotsAndBoxesEngine,
    tt: TranspositionTable,
    depth: int,
    root_player: int,
) -> list[Action]:
    """Return actions unchanged (baseline ordering)."""

    return actions


def _completes_box(board: BitBoard, action: Action) -> bool:
    """Return True if the action would complete any adjacent box."""

    for box in get_boxes_adjacent_to_edge(action):
        if bit_count_box_edges(board, box) == 3:
            return True
    return False


def move_ordering(
    actions: list[Action],
    eng: DotsAndBoxesEngine,
    tt: TranspositionTable,
    depth: int,
    root_player: int,
) -> list[Action]:
    """Prioritize forced moves, then safe moves, then the rest."""

    edges = eng.get_state().board

    forced = []
    safe = []
    for a in actions:
        if _completes_box(edges, a):
            forced.append(a)
        if not bit_makes_third_edge(edges, a):
            safe.append(a)

    rest = [a for a in actions if a not in forced and a not in safe]
    return forced + safe + rest


def move_ordering_v2(
    actions: list[Action],
    eng: DotsAndBoxesEngine,
    depth: int,
    root_player: int,
) -> list[Action]:
    """Alternate ordering including double-cross preference."""

    edges = eng.get_state().board

    forced: list[Action] = []
    double_cross: list[Action] = []
    safe: list[Action] = []

    for a in actions:
        if _completes_box(edges, a):
            forced.append(a)
        if _makes_double_cross(edges, a):
            double_cross.append(a)
        if not bit_makes_third_edge(edges, a):
            safe.append(a)

    rest = [a for a in actions if a not in forced and a not in double_cross and a not in safe]
    return forced + safe + double_cross + rest


def _makes_double_cross(board: BitBoard, action: Action) -> bool:
    """Return True if the action creates two boxes each with two edges."""

    boxes = get_boxes_adjacent_to_edge(action)
    if len(boxes) != 2:
        return False
    return all(bit_count_box_edges(board, box) == 2 for box in boxes)
