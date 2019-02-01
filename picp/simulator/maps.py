import collections
from math import cos, sin

from picp.util.pose import Pose
from picp.util.position import Position


class Wall:
    def __init__(self, p1: Position, p2: Position):
        self.p1 = p1
        self.p2 = p2

    def move(self, x, y):
        m = Position.from_list([x, y])
        self.p1 += m
        self.p2 += m
        return self

    def flix_x(self):
        self.p1 = self.p1.flip_x()
        self.p2 = self.p2.flip_x()
        return self

    def copy(self):
        return Wall(self.p1.copy(), self.p2.copy())

    def __str__(self):
        return f"p1:{self.p1}, p2:{self.p2}"

    def __hash__(self):
        return hash(str(self))


def create_basic_hallway(orientation=0):
    return [Wall(Position(-2, 6).rotate(orientation), Position(-2, -6).rotate(orientation)),
            Wall(Position(+2, 6).rotate(orientation), Position(+2, -6).rotate(orientation))]


def create_bumpy_hallway():
    step_x = 0.5
    step_y = 1
    hallway_width = 3
    walls_template = [Wall(Position(hallway_width, 0), Position(hallway_width + step_x, 0)),
                      Wall(Position(hallway_width, 0), Position(hallway_width, step_y)),
                      Wall(Position(hallway_width, step_y), Position(hallway_width + step_x, step_y)),
                      Wall(Position(hallway_width + step_x, step_y), Position(hallway_width + step_x, 2*step_y))]
    walls = []
    walls += walls_template
    walls += [w.copy().move(0, 4*step_y) for w in walls_template]
    walls += [w.copy().move(0, 2*step_y) for w in walls_template]
    walls += [w.copy().move(0, -2*step_y) for w in walls_template]
    walls += [w.copy().move(0, -4*step_y) for w in walls_template]

    walls += [w.copy().flix_x() for w in walls]
    return walls


WALL = 'X'
ORIGIN = '*'
NUMBERS = [str(i) for i in range(1, 10)]

def print_grid(grid):
    for l in grid:
        for c in l:
            print(c, end='')
        print()

def find_shortest_path(ascii_art, a, b):
    # Basic burning bush pathfinder
    for n in NUMBERS:
        ascii_art = ascii_art.replace(n, '.')

    grid = [[c for c in l] for l in ascii_art.splitlines()]

    cardinals = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    to_visit = set([b])
    visited = set([])
    cost = 0
    solved = False
    while len(to_visit) > 0 and not solved:
        new_to_visit = set([])
        for x, y in to_visit:
            if (x, y) == a:
                solved = True
                break
            grid[y][x] = cost
            for dx, dy in cardinals:
                p = dx + x, dy + y
                if grid[p[1]][p[0]] != WALL and p not in visited and p not in to_visit:
                    new_to_visit.add(p)
            visited.add((x, y))
        to_visit = list(new_to_visit)
        cost +=1

    # Always take the shortest tile until you reach the destination
    curr = a
    path = []
    solved = False
    while not solved:
        x, y = curr
        next_curr = None
        for dx, dy in cardinals:
            p = dx + x, dy + y
            px, py = p
            p_cost = grid[py][px]
            if cost is None or (isinstance(p_cost, int) and p_cost < cost):
                cost = p_cost
                next_curr = p
            if (px, py) == b:
                solved = True
                break
        path.append(next_curr)
        curr = next_curr
    # for x, y in path:
    #     grid[y][x] = '*'
    # print_grid(grid)
    # print(path)
    # print()
    return path


def subsample_path(path):
    # Add a point between each point in path
    for (ax, ay), (bx, by) in zip(path, path[1:]):
        yield (ax, ay)
        yield (0.5 * (bx - ax) + ax,
               0.5 * (by - ay) + ay)
    yield path[-1]


def from_ascii_art(ascii_art, origin_offset=None, orientation=0):
    print(ascii_art)
    if origin_offset is None:
        origin_offset = Position()
    grid = [[c for c in l] for l in ascii_art.splitlines()]
    cardinals = [(-1, 0), (0, -1)]
    w = len(grid[0])
    h = len(grid)
    cells = []
    origin = None
    checkpoints = {}
    for y, l in enumerate(grid):
        for x, val in enumerate(l):
            is_free = val != WALL
            if val == ORIGIN:
                origin = x, y
                checkpoints[0] = origin
            if val in NUMBERS:
                checkpoints[int(val)] = x, y
            for cx, cy in cardinals:
                px = cx + x
                py = cy + y
                if 0 <= px < w and 0 <= py < h:
                    is_free_c = grid[py][px] != WALL
                    if is_free != is_free_c:
                        # print(f"Add wall at {x},{y} in {cx},{cy}")
                        cells.append((x, y, cy == 0))

    if origin is None:
        raise RuntimeError("No origin found, missing the '*' in map")

    ox, oy = origin

    def to_real_world(cell_x, cell_y, offset=0.4):
        ax = (cell_x - ox - offset) * cell_w
        ay = (cell_y - oy - offset) * cell_h
        x = ax * cos(orientation) - ay * sin(orientation)
        y = ax * sin(orientation) + ay * cos(orientation)
        return x + origin_offset.x, y + origin_offset.y

    cell_w = 3
    cell_h = -3  # Matplotlib has the origin in the lower left corner, not the upper left conner
    walls = []
    for x, y, is_vertical in cells:
        if is_vertical:
            w = Wall(Position(*to_real_world(x, y)), Position(*to_real_world(x, y + 1)))
        else:
            w = Wall(Position(*to_real_world(x, y)), Position(*to_real_world(x + 1, y)))
        walls.append(w)

    if len(checkpoints) > 1:
        ordered_checkpoints = list(collections.OrderedDict(sorted(checkpoints.items())).values())
        # linear_path += [origin]  # So it loop
        path = [origin]
        for a, b in zip(ordered_checkpoints, ordered_checkpoints[1:]):
            path += find_shortest_path(ascii_art, a, b)
    path = subsample_path(path)
    # path = subsample_path(list(path))
    pose_path = [Pose.from_values(*to_real_world(x, y, offset=0.0), orientation) for x, y in path]
    return walls, pose_path