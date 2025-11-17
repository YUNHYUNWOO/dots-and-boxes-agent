import pygame
import sys
from typing import Optional, Tuple

# ---------------------------
# Config
# ---------------------------
GRID_SIZE = 5          # number of boxes per side (NxN). Dots are (N+1)x(N+1)
SPACING = 90           # pixels between adjacent dots
MARGIN = 80            # outer margin
DOT_RADIUS = 6
LINE_THICK = 8
HITBOX_PAD = 14        # extra thickness for click hitboxes

BG_COLOR = (24, 26, 27)
DOT_COLOR = (230, 232, 234)
LINE_COLOR_P1 = (92, 193, 255)
LINE_COLOR_P2 = (255, 149, 112)
LINE_COLOR_NEUTRAL = (120, 130, 140)
BOX_FILL_P1 = (92, 193, 255, 60)
BOX_FILL_P2 = (255, 149, 112, 60)
TEXT_COLOR = (230, 232, 234)
HOVER_COLOR = (180, 180, 180)
WIN_MSG_COLOR = (255, 230, 90)

FPS = 60

# ---------------------------
# Helpers
# ---------------------------
Edge = Tuple[str, int, int]  # (orientation 'H'/'V', row, col)


def grid_to_px(r: int, c: int) -> Tuple[int, int]:
    x = MARGIN + c * SPACING
    y = MARGIN + r * SPACING
    return x, y

def compute_window(nboxes: int) -> Tuple[int, int]:
    width = MARGIN * 2 + SPACING * nboxes
    height = MARGIN * 2 + SPACING * nboxes + 150
    return width, height

def get_render_desc(n_box):
    screen = pygame.display.set_mode(compute_window(n_box))
    clock = pygame.time.Clock()
    fonts = {
        'xl': pygame.font.SysFont("arial", 40, bold=True),
        'lg': pygame.font.SysFont("arial", 28, bold=True),
        'md': pygame.font.SysFont("arial", 24, bold=True),
        'sm': pygame.font.SysFont("arial", 18)
    }

    return screen, clock, fonts

def edge_rect(edge: Edge) -> pygame.Rect:
    ori, r, c = edge
    if ori == 'H':
        x1, y1 = grid_to_px(r, c)
        x2, _ = grid_to_px(r, c + 1)
        cx = (x1 + x2) // 2
        w = (x2 - x1)
        h = LINE_THICK + HITBOX_PAD
        return pygame.Rect(cx - w // 2, y1 - h // 2, w, h)
    else:
        x1, y1 = grid_to_px(r, c)
        _, y2 = grid_to_px(r + 1, c)
        cy = (y1 + y2) // 2
        h = (y2 - y1)
        w = LINE_THICK + HITBOX_PAD
        return pygame.Rect(x1 - w // 2, cy - h // 2, w, h)




def neighbors_of_edge(edge: Edge, n: int):
    """Return list of (box_r, box_c) that this edge borders (1 or 2 boxes)."""
    ori, r, c = edge
    boxes = []
    if ori == 'H':
        # Above box at (r-1, c) and below box at (r, c)
        if 0 <= r - 1 < n:
            boxes.append((r - 1, c))
        if 0 <= r < n:
            boxes.append((r, c))
    elif ori=='V':
        # Left box at (r, c-1) and right box at (r, c)
        if 0 <= c - 1 < n:
            boxes.append((r, c - 1))
        if 0 <= c < n:
            boxes.append((r, c))
    return boxes


class DotsAndBoxes:
    def __init__(self, n: int):
        self.n = n
        # Track claimed edges: None (unclaimed) or 0/1 (player index)
        self.h_edges = [[None for _ in range(n)] for __ in range(n + 1)]
        self.v_edges = [[None for _ in range(n + 1)] for __ in range(n)]
        # Boxes owners: None or 0/1
        self.box_owner = [[None for _ in range(n)] for __ in range(n)]
        self.current_player = 0
        self.score = [0, 0]
        self.total_boxes = n * n
        self.hover_edge: Optional[Edge] = None
        self.is_game_over = False

        # Precompute all edges + hitboxes for faster hit-testing
        self.all_edges: list[Edge] = []
        self.hitboxes: list[pygame.Rect] = []
        for r in range(n + 1):
            for c in range(n):
                e = ('H', r, c)
                self.all_edges.append(e)
                self.hitboxes.append(edge_rect(e))
        for r in range(n):
            for c in range(n + 1):
                e = ('V', r, c)
                self.all_edges.append(e)
                self.hitboxes.append(edge_rect(e))

    def edge_claimed(self, e: Edge) -> bool:
        ori, r, c = e
        if ori == 'H':
            return self.h_edges[r][c] is not None
        elif ori == 'V':
            return self.v_edges[r][c] is not None

    def turn_over(self):
        self.current_player = 0 if self.current_player == 1 else 1
        
    def claim_edge(self, e: Edge) -> bool:
        """ Attempt to claim edge for current player. 
            Returns False if edge is already claimed

            Else Returns made_box, box_idx
        """
        if self.edge_claimed(e):
            return False
        ori, r, c = e
        
        if ori == 'H':
            self.h_edges[r][c] = self.current_player
        elif ori == 'V':
            self.v_edges[r][c] = self.current_player

        made_box = False
        box_idx = []
        ## Undo를 위해서는 
        ## 어떤 박스가 생겼는지
        ## Box index를 기록해야한다.
        for (br, bc) in neighbors_of_edge(e, self.n):
            if self.box_owner[br][bc] is None:
                if self.is_box_complete(br, bc):
                    self.box_owner[br][bc] = self.current_player
                    self.score[self.current_player] += 1

                    box_idx.append([br, bc])
                    made_box = True


        if not made_box:
            self.turn_over()

        if sum(self.score) == self.total_boxes:
            self.is_game_over = True

        return  {
            'made_box': made_box, 
            'box_idx': box_idx, 
        }

    def is_box_complete(self, r: int, c: int) -> bool:
        # Box at (r,c) has edges: H(r,c), H(r+1,c), V(r,c), V(r,c+1)
        return (
            self.h_edges[r][c] is not None and
            self.h_edges[r + 1][c] is not None and
            self.v_edges[r][c] is not None and
            self.v_edges[r][c + 1] is not None
        )

    def game_over(self) -> bool:
        return self.is_game_over


    def winner(self) -> Optional[int]:
        if not self.game_over():
            return None
        if self.score[0] > self.score[1]:
            return 0
        if self.score[1] > self.score[0]:
            return 1
        return -1  # tie


    def find_hover_edge(self, pos: Tuple[int, int]) -> Optional[Edge]:
        x, y = pos
        # Quick reject outside bounding board area
        board_rect = pygame.Rect(MARGIN - 20, MARGIN - 20,
                                 self.n * SPACING + 40, self.n * SPACING + 40)
        if not board_rect.collidepoint(x, y):
            return None
        # Iterate through hitboxes: choose the closest unclaimed under the cursor
        for e, rect in zip(self.all_edges, self.hitboxes):
            if rect.collidepoint(x, y) and not self.edge_claimed(e):
                return e
        return None



def draw_board(screen, game: DotsAndBoxes, fonts):

    screen.fill(BG_COLOR)
    # Box fills (semi-transparent)
    box_surface = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for r in range(game.n):
        for c in range(game.n):
            owner = game.box_owner[r][c]
            if owner is None:
                continue
            x1, y1 = grid_to_px(r, c)
            x2, y2 = grid_to_px(r + 1, c + 1)
            rect = pygame.Rect(x1 + LINE_THICK//2, y1 + LINE_THICK//2,
                            (x2 - x1) - LINE_THICK, (y2 - y1) - LINE_THICK)
            color = BOX_FILL_P1 if owner == 0 else BOX_FILL_P2
            pygame.draw.rect(box_surface, color, rect)
    screen.blit(box_surface, (0, 0))
    # Draw claimed edges
    # Horizontal
    for r in range(game.n + 1):
        for c in range(game.n):
            owner = game.h_edges[r][c]
            x1, y1 = grid_to_px(r, c)
            x2, _ = grid_to_px(r, c + 1)
            if owner is not None:
                color = LINE_COLOR_P1 if owner == 0 else LINE_COLOR_P2
                pygame.draw.line(screen, color, (x1, y1), (x2, y1), LINE_THICK)
            else:
                pygame.draw.line(screen, (60, 65, 70), (x1, y1), (x2, y1), 2)
    # Vertical
    for r in range(game.n):
        for c in range(game.n + 1):
            owner = game.v_edges[r][c]
            x1, y1 = grid_to_px(r, c)
            _, y2 = grid_to_px(r + 1, c)
            if owner is not None:
                color = LINE_COLOR_P1 if owner == 0 else LINE_COLOR_P2
                pygame.draw.line(screen, color, (x1, y1), (x1, y2), LINE_THICK)
            else:
                pygame.draw.line(screen, (60, 65, 70), (x1, y1), (x1, y2), 2)
    # Hover edge highlight
    if game.hover_edge is not None and not game.edge_claimed(game.hover_edge):
        e = game.hover_edge
        ori, r, c = e
        if ori == 'H':
            x1, y1 = grid_to_px(r, c)
            x2, _ = grid_to_px(r, c + 1)
            pygame.draw.line(screen, HOVER_COLOR, (x1, y1), (x2, y1), LINE_THICK)
        else:
            x1, y1 = grid_to_px(r, c)
            _, y2 = grid_to_px(r + 1, c)
            pygame.draw.line(screen, HOVER_COLOR, (x1, y1), (x1, y2), LINE_THICK)
    # Dots on top
    for r in range(game.n + 1):
        for c in range(game.n + 1):
            x, y = grid_to_px(r, c)
            pygame.draw.circle(screen, DOT_COLOR, (x, y), DOT_RADIUS)
    # Sidebar / Score
    p1_col = LINE_COLOR_P1
    p2_col = LINE_COLOR_P2
    score_text = fonts['lg'].render(f"P1: {game.score[0]}    P2: {game.score[1]}", True, TEXT_COLOR)
    screen.blit(score_text, (MARGIN, MARGIN + game.n * SPACING + 36))
    turn_color = p1_col if game.current_player == 0 else p2_col
    turn_text = fonts['md'].render(f"Turn: P{game.current_player + 1}", True, turn_color)
    screen.blit(turn_text, (MARGIN, MARGIN + game.n * SPACING + 72))
    hint_text = fonts['sm'].render("Click edges to draw. R = restart, +/- = grid size, Esc = quit", True, (180, 180, 185))
    screen.blit(hint_text, (MARGIN, MARGIN + game.n * SPACING + 104))
    # Game over banner
    if game.is_game_over:
        w = screen.get_width()
        banner = pygame.Surface((w, 70), pygame.SRCALPHA)
        banner.fill((0, 0, 0, 140))
        screen.blit(banner, (0, 10))
        winner = game.winner()
        if winner == -1:
            msg = "Game Over — Tie!"
        else:
            msg = f"Game Over — Winner: P{winner + 1}!"
        txt = fonts['xl'].render(msg, True, WIN_MSG_COLOR)
        screen.blit(txt, (MARGIN, 24))




# ---------------------------
# Main loop
# ---------------------------

def main():
    pygame.init()
    pygame.display.set_caption("Dots and Boxes (pygame)")

    n = GRID_SIZE

    def compute_window(nboxes: int) -> Tuple[int, int]:
        width = MARGIN * 2 + SPACING * nboxes
        height = MARGIN * 2 + SPACING * nboxes + 150
        return width, height

    screen, clock, fonts = get_render_desc(n)

    game = DotsAndBoxes(n)

    running = True
    while running:
        clock.tick(FPS)
        mouse_pos = pygame.mouse.get_pos()
        game.hover_edge = game.find_hover_edge(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game = DotsAndBoxes(n)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):  # '+' on many keyboards
                    n = min(9, n + 1)
                    screen = pygame.display.set_mode(compute_window(n))
                    game = DotsAndBoxes(n)
                elif event.key == pygame.K_MINUS:
                    n = max(2, n - 1)
                    screen = pygame.display.set_mode(compute_window(n))
                    game = DotsAndBoxes(n)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if game.is_game_over:
                    # On game over, left click restarts
                    game = DotsAndBoxes(n)
                else:
                    e = game.find_hover_edge(event.pos)
                    if e is not None:
                        game.claim_edge(e)

        draw_board(screen, game, fonts)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
