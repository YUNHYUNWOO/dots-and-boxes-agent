"""Human-playable policy that waits for a player to click an edge."""

import pygame

from dotsandboxes import DotsAndBoxes, DnBEnv
from util.time_manager import TimeManager

from .basepolicy import BasePolicy


class PlayablePolicy(BasePolicy):
    def __init__(self):
        super().__init__()

    def get_action(self, observation, info, env: DnBEnv, time_manager: TimeManager):
        """Return the edge selected by the human player via a mouse click."""

        dnb: DotsAndBoxes = env.DnB
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    edge = dnb.find_hover_edge(event.pos)
                    if edge is not None:
                        waiting = False
                        break
        direction, row, col = edge
        return [col, row, 0 if direction == 'H' else 1]
