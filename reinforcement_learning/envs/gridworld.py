from typing import Tuple

class Gridworld:
    """
    States are (row, col). Actions: 0=up, 1=down, 2=left, 3=right.
    Episode ends on goal, pit, or max_steps.
    Rewards: step -1, goal +20, pit -10.
    """
    def __init__(self, rows: int = 5, cols: int = 5, max_steps: int = 100):
        self.rows, self.cols = rows, cols
        self.max_steps = max_steps

        # layout — tweak for your report if you like
        self.start = (0, 0)
        self.goal  = (4, 4)
        self.pits  = {(2, 2)}
        self.walls = {(1, 2), (2, 1)}

        self.action_space = 4
        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.s = self.start
        self.steps = 0
        return self.s

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def step(self, a: int):
        r, c = self.s
        moves = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}  # up, down, left, right
        dr, dc = moves[a]
        nr, nc = r + dr, c + dc

        # clamp to grid; bump into walls/bounds → stay
        if not self.in_bounds(nr, nc) or (nr, nc) in self.walls:
            nr, nc = r, c

        s_next = (nr, nc)
        self.steps += 1

        reward, done = -1.0, False
        if s_next in self.pits:
            reward, done = -10.0, True
        elif s_next == self.goal:
            reward, done = +20.0, True
        elif self.steps >= self.max_steps:
            done = True

        self.s = s_next
        return s_next, reward, done, {}

    def render(self, policy=None) -> None:
        """Print an ASCII grid. Optionally pass a (rows, cols) policy with actions {0,1,2,3}."""
        arrows = {0:'↑', 1:'↓', 2:'←', 3:'→'}
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                p = (r, c)
                if p == self.s:
                    row_cells.append('A')
                elif p == self.start:
                    row_cells.append('S')
                elif p == self.goal:
                    row_cells.append('G')
                elif p in self.pits:
                    row_cells.append('X')
                elif p in self.walls:
                    row_cells.append('#')
                else:
                    row_cells.append(arrows[int(policy[r, c])] if policy is not None else '.')
            print(' '.join(row_cells))
        print()
